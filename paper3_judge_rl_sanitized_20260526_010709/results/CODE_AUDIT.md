# Paper 3 (Judge RL) — Comprehensive Code Audit

**Date**: 2026-05-17
**Auditor**: Code Audit Agent
**Scope**: train_judge_grpo.py, eval_judge.py, prepare_data.py, create_length_confounded.py, data files

---

## Executive Summary

Found **2 CRITICAL**, **4 WARNING**, **8 OK** findings. The two critical issues are:

1. **Unbalanced training experiments used 100%-A gold labels** — all EXP-006/007a/008/009 (non-"b" variants) trained on `judge_train.json` where gold_label is always "A". The reward function literally rewards "always say A". These models' accuracy gains are confounded with position bias amplification.

2. **Eval accuracy metric is degenerate for all-A gold test data** — `rewardbench_test.json` has gold_label = "A" for all 449 samples. Accuracy == pred_A_rate EXACTLY. This is not technically wrong (RewardBench chosen IS in position A), but makes it impossible to distinguish "model learned quality judgment" from "model learned to always say A".

No CRITICAL bugs in the code logic itself (reward computation, parsing, swap logic are all correct). The issues are in the experimental design and data pipeline.

---

## File-by-File Audit

### 1. train_judge_grpo.py

#### 1.1 parse_judge_output (lines 61-84)
**Status**: 🟡 WARNING — edge case crash

The regex `r'\[\[(A|B|C),?\s*([\d.]+)\]\]'` correctly extracts choice + confidence. However:
- Input `[[A, .]]` (dot only, no digits) causes `float('.')` → **ValueError crash**
- Input `[[A0.8]]` (no comma) matches and parses as `choice='A', confidence=0.8` — this is accidental but harmless since `0.8` is the default anyway

**Impact**: Low — unlikely to occur in practice since the model rarely outputs `[[A, .]]`. But should be hardened with a try/except around `float()`.

#### 1.2 compute_accuracy_reward (lines 87-90)
**Status**: 🟢 OK

Simple `parsed["choice"] == gold_label` comparison. Works correctly for both A and B gold labels. No issues.

#### 1.3 compute_consistency_reward (lines 93-102)
**Status**: 🟢 OK (with caveat)

The flip logic `A↔B, C↔C` is correct. But this function is **never called during training** — it's defined but unused. The actual training uses a "decisiveness" proxy inline (line 238, 247). This is a known design decision, documented in REVIEW_AND_FIX.md.

#### 1.4 compute_calibration_reward (lines 105-110)
**Status**: 🟡 WARNING — calibration is fake

Brier score formula `(confidence - correct)^2` is mathematically correct. But the confidence values are hardcoded defaults (`0.8` for A/B, `0.5` for C) when the model doesn't output explicit confidence. The prompt template does NOT ask for confidence values, so almost all outputs will use the default.

**Impact**: The "calibration reward" effectively degenerates to:
- Correct → 1 - (0.8 - 1.0)^2 = 0.96
- Wrong → 1 - (0.8 - 0.0)^2 = 0.36
- Tie → 1 - (0.5 - 0/1)^2 = 0.75

This is just a rescaled accuracy reward with a tie bonus. Known issue documented in REVIEW_AND_FIX.md. **If the paper claims calibration improvement, this is misleading.**

#### 1.5 reward_function (lines 213-256)
**Status**: 🟢 OK

The function signature `def reward_function(completions, gold_label, **kwargs)` is compatible with TRL's GRPOTrainer API. The indexing `gold_label[i]` corresponds correctly to `completions[i]` because TRL replicates dataset columns for each generation in the group.

#### 1.6 Gold label alignment with GRPO groups
**Status**: 🟢 OK

TRL GRPOTrainer automatically replicates the `gold_label` column for each of the `num_generations` (group_size=8) completions per prompt. So `gold_label[i]` correctly matches `completions[i]`. No index misalignment.

#### 1.7 Default training data path
**Status**: 🔴 CRITICAL — 100% A gold labels in default data

Default `--train_data` points to `judge_train.json` which has **100% gold_label="A"** (2089/2089). The reward function gives reward=1.0 for predicting A and reward=0.0 for B. This trains the model to **always say A**, not to judge quality.

**Affected experiments**: ALL non-"b" variants:
- EXP-006, EXP-006_s2, EXP-006_s3
- EXP-007a, EXP-007a_s2
- EXP-008, EXP-008_s2
- EXP-009, EXP-009_s2, EXP-009_s3, EXP-009_s4
- EXP-009_lr1e5, EXP-009_lr1e6
- pilot

**Unaffected experiments** (used balanced data): EXP-006b*, EXP-007b*, EXP-008b*, EXP-009b*, balanced_*

**Evidence from eval results**: Accuracy == pred_A_rate exactly:
| Model | Accuracy | pred_A% | pred_B% |
|-------|----------|---------|---------|
| Baseline | 80.2% | 80.2% | 15.8% |
| EXP-006 (100%A train) | 94.4% | 94.4% | 4.5% |
| EXP-009 lr=1e-5 (100%A) | 98.9% | 98.9% | 0.7% |

The lr=1e-5 model says A 98.9% of the time — it learned pure position bias.

#### 1.8 Seed setting (lines 140-144)
**Status**: 🟢 OK

`random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()` are all set. Also passed to `GRPOConfig(seed=args.seed)`. Seeds work correctly.

---

### 2. eval_judge.py

#### 2.1 parse_judge_output (lines 70-89)
**Status**: 🟢 OK

Same regex as training script. Eval version returns `PARSE_FAIL` instead of defaulting to `C` — this is better for eval since it allows counting parse failures. Same `[[A, .]]` edge case exists but is very low probability.

#### 2.2 Consistency calculation (lines 312-315)
**Status**: 🟢 OK

```python
flip_map = {"A": "B", "B": "A", "C": "C", "PARSE_FAIL": "PARSE_FAIL"}
expected_swap = flip_map.get(parsed_orig["choice"], "PARSE_FAIL")
is_consistent = parsed_swap["choice"] == expected_swap
```

This is correct. If original predicts A, swap should predict B (positions are flipped). Handles PARSE_FAIL correctly (PARSE_FAIL is never consistent with anything).

#### 2.3 Accuracy vs gold_label
**Status**: 🔴 CRITICAL (evaluation design, not code bug)

Eval uses `gold = item["gold_label"]` from `rewardbench_test.json` where gold is always "A" (449/449). Therefore:
- `is_correct = (predicted == "A")` for every sample
- Accuracy = pred_A_rate

This is **technically correct** (chosen IS in position A) but **methodologically problematic** because:
- For unbalanced-trained models: high "accuracy" just means high position-A bias
- Cannot distinguish quality judgment from position shortcut
- Paper cannot claim "accuracy improved" without swap-corrected accuracy

**Recommended fix**: Report **swap-corrected accuracy** = (accuracy_on_orig + accuracy_on_swap) / 2, which is immune to position bias. The eval script already evaluates on swap data — just need to also compute accuracy against swap gold_label.

#### 2.4 LoRA adapter loading (lines 262-265)
**Status**: 🟢 OK

```python
model = PeftModel.from_pretrained(model, args.adapter_path)
model = model.merge_and_unload()
```

Correct. Loads adapter, merges into base weights, removes PEFT wrapper. All layers are merged (no partial loading risk with `merge_and_unload()`).

#### 2.5 Batch generation (lines 92-127)
**Status**: 🟢 OK

Left-padding + slicing at `inputs['input_ids'].shape[1]` correctly extracts new tokens. Temperature=0.1 with `do_sample=True` is appropriate for eval (near-deterministic but not greedy).

#### 2.6 Brier score computation (lines 318-320)
**Status**: 🟢 OK

Formula is correct: `brier = (conf - correct_binary)^2`. Uses parsed confidence from model output. The same fake-confidence caveat applies if model doesn't output explicit confidence.

---

### 3. prepare_data.py

#### 3.1 build_judge_instance (lines 53-78)
**Status**: 🟢 OK

Position swap logic is correct:
- `swap=False`: chosen→A, rejected→B, gold="A"
- `swap=True`: rejected→A, chosen→B, gold="B"

The prompt template is correctly formatted with the swapped responses in swapped positions. Verified empirically: all 20 checked pairs have different prompts, and answers are correctly swapped.

#### 3.2 RewardBench field mapping
**Status**: 🟢 OK

Fields `prompt`, `chosen`, `rejected` are correct for RewardBench. The `instruction` fallback in `.get()` is harmless dead code.

#### 3.3 Train data generates 100%-A gold for non-swap
**Status**: 🟡 WARNING (design, not bug)

`judge_train.json` contains only `swap=False` instances → all gold="A". The `judge_swap.json` is a separate file with all gold="B". This is correct data preparation, BUT the training script defaults to only `judge_train.json`, creating the 100%-A confound.

The balanced data (`judge_train_balanced.json`) correctly interleaves original and swap samples: 2089 gold=A + 2089 gold=B = 4178 total, 50/50 distribution. This is the correct data to train on.

---

### 4. create_length_confounded.py

#### 4.1 extract_response_lengths (lines 21-50)
**Status**: 🟢 OK

Regex correctly extracts text between `[The Start of Assistant X's Answer]` and `[The End of Assistant X's Answer]`. Uses `re.DOTALL` for multiline. Returns character count (`.strip()` removes whitespace padding).

#### 4.2 Label assignment (lines 83-91)
**Status**: 🟢 OK

Correctly assigns gold_label based on which response is longer:
- A longer → gold="A"
- B longer → gold="B"
- Equal → skipped

This creates a dataset where gold label = longer response, which is the intended confound.

#### 4.3 Output data distribution
**Status**: 🟢 OK

Verified: 1840 A + 1840 B = 3680 total. 50/50 distribution (naturally, since balanced data has 50/50 original/swap, and length doesn't correlate with position).

---

### 5. Data File Verification

#### 5.1 judge_train.json
**Status**: 🟡 WARNING

- 2089 samples, **100% gold_label="A"**
- All `swapped=False`
- This is the DEFAULT training data and causes the position bias confound

#### 5.2 judge_train_balanced.json
**Status**: 🟢 OK

- 4178 samples, **50.0% A, 50.0% B**
- 2089 swapped=True, 2089 swapped=False
- Verified 20 pairs: all have DIFFERENT prompts (positions correctly swapped)
- Verified 5 pairs: answers correctly swapped (orig_A == swap_B and vice versa)
- Gold labels correctly flipped (orig: A→A, swap: same-pair→B)

#### 5.3 judge_swap.json
**Status**: 🟢 OK

- 2089 samples, aligned with judge_train.json (0 misaligned pairs)
- Same original_id ordering

#### 5.4 rewardbench_test.json / rewardbench_test_swap.json
**Status**: 🟢 OK (with design caveat)

- test: 449 samples, **100% gold_label="A"**
- test_swap: 449 samples, **100% gold_label="B"**
- Verified 10 pairs: answers correctly swapped
- Correctly aligned by index

The all-A gold in test data is inherent to RewardBench design (chosen is always in position A for non-swapped). This is not a data bug but creates the eval confound noted in section 2.3.

#### 5.5 judge_train_augmented.json
**Status**: 🟢 OK

- 4178 samples, 50.0% A, 50.0% B
- Appears identical to balanced data (same structure)

---

## Impact Assessment

### On the Paper's Core Claim: "RL Training Destroys Position Consistency"

The claim is **VALID but CONFOUNDED for unbalanced experiments**:

**Unbalanced (100%A) experiments**: The consistency collapse (83% → 38-62%) is real, but the mechanism is trivially explained: the model learned "always say A" from training data that rewards only A. On original prompt, it says A (correct). On swapped prompt, it still says A (incorrect, should flip to B). This isn't a deep insight about RL — it's a data artifact.

**What would validate the claim**: The BALANCED experiments (EXP-006b, 007b, 008b, 009b) trained on 50/50 data. If consistency STILL drops for balanced models, that's the real finding. **But none of the balanced models have been evaluated yet** (no eval_results.json files found).

### Affected Results

| Experiment Group | Train Data | Gold Distribution | Position Bias Confound? | Eval Done? |
|-----------------|-----------|-------------------|------------------------|-----------|
| EXP-006/007a/008/009 (unbalanced) | judge_train.json | 100% A | YES - results unreliable | Yes |
| EXP-006b/007b/008b/009b (balanced) | judge_train_balanced.json | 50% A / 50% B | No confound | **NO - NOT EVALUATED** |
| EXP-LENGTH (confounded) | judge_train_length_confounded.json | 50% A / 50% B | No position confound | Unclear |
| Baseline | N/A | N/A | N/A | Yes |

---

## Recommendations

### Immediate (before any paper claims)

1. **Evaluate ALL balanced models** — these are the only unconfounded results
2. **Compute swap-corrected accuracy** for all models: `(acc_orig + acc_swap) / 2` where `acc_swap = (predicted_on_swap == swap_gold_label) / total`
3. **Report pred_A rate** alongside accuracy to make position bias transparent

### Code Fixes

4. **Change default `--train_data`** in `train_judge_grpo.py` to `judge_train_balanced.json`
5. **Add try/except** around `float(match.group(2))` in parse_judge_output for robustness
6. **Add swap-corrected accuracy** to `compute_metrics()` in eval_judge.py

### Paper Narrative

7. The unbalanced experiments are actually **interesting as a control**: they show that RL with position-confounded data amplifies position bias (expected). The balanced experiments are the real test of whether RL intrinsically creates position bias.
8. If balanced models ALSO show consistency drop: strong paper ("RL intrinsically harms consistency")
9. If balanced models maintain consistency: different but still valid paper ("data confounds drive bias in RL judges")

---

## Summary Table

| # | File | Finding | Severity |
|---|------|---------|----------|
| 1 | train_judge_grpo.py:132 | Default train data is 100%-A gold | 🔴 CRITICAL |
| 2 | eval_judge.py:309-310 | Eval accuracy == pred_A_rate (all-A gold test) | 🔴 CRITICAL |
| 3 | train_judge_grpo.py:69-73 | parse_judge_output crashes on `[[A, .]]` | 🟡 WARNING |
| 4 | train_judge_grpo.py:105-110 | Calibration uses fake confidence (hardcoded 0.8/0.5) | 🟡 WARNING |
| 5 | train_judge_grpo.py:238 | "Consistency" reward is actually decisiveness proxy | 🟡 WARNING |
| 6 | prepare_data.py | 100%-A gold in judge_train.json (by design) | 🟡 WARNING |
| 7 | train_judge_grpo.py:87-90 | Accuracy reward logic | 🟢 OK |
| 8 | train_judge_grpo.py:93-102 | Consistency flip logic | 🟢 OK |
| 9 | eval_judge.py:312-315 | Eval consistency flip logic | 🟢 OK |
| 10 | eval_judge.py:92-127 | Batch generation padding/truncation | 🟢 OK |
| 11 | eval_judge.py:262-265 | LoRA adapter merge | 🟢 OK |
| 12 | prepare_data.py:53-78 | Prompt swap construction | 🟢 OK |
| 13 | create_length_confounded.py | Length-based label assignment | 🟢 OK |
| 14 | Data: judge_train_balanced.json | 50/50 distribution, swaps verified | 🟢 OK |
