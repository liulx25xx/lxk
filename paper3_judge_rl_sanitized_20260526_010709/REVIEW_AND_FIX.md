# Paper 3 (Judge RL) — Experiment Audit Report

**Date**: 2026-05-17  
**Auditor**: ML Experiment Audit Agent

---

## Executive Summary

Identified **4 critical**, **3 major**, and **3 minor** issues. The most severe are:
1. Training script is DRY RUN only (trainer code commented out)
2. Reward function signature incompatible with TRL GRPOTrainer API
3. Consistency reward design is fundamentally unresolvable within single GRPO pass
4. Calibration reward uses fake confidence (always 0.8/0.5), not model logprobs

---

## Issues (Sorted by Severity)

### CRITICAL-1: Training Script is Dry Run Only

**File**: `scripts/train_judge_grpo.py:222-232`

The actual GRPOTrainer instantiation and `.train()` call are **commented out** with a `# TODO: Uncomment below when ready`. The script currently just prints "DRY RUN" and exits.

**Fix**: Uncomment and correct the trainer instantiation (after fixing the API issues below).

---

### CRITICAL-2: Reward Function Signature Incompatible with TRL GRPOTrainer

**File**: `scripts/train_judge_grpo.py:180-210`

Current signature:
```python
def reward_function(completions, prompts, **kwargs):
```

TRL GRPOTrainer (>=1.0) expects:
```python
def reward_function(completions, **kwargs):
```

Where:
- `completions` is `list[list[dict[str, str]]]` (message format), NOT a list of strings
- Extra dataset columns (e.g., `gold_label`) are passed via `**kwargs`
- Must return `list[float]`

The current code treats `completions` as a list of strings and indexes `gold_labels` with `i % len(gold_labels)` which is completely wrong — the trainer batches and flattens across samples × group_size.

**Fix**: Rewrite reward function with correct signature:
```python
def reward_function(completions, gold_label, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"]  # Extract text from message format
        gold = gold_label[i]
        # ... compute reward on text vs gold
        rewards.append(reward_value)
    return rewards
```

---

### CRITICAL-3: Consistency Reward Cannot Be Computed Within Standard GRPO

**File**: `scripts/train_judge_grpo.py:80-89` and `train_judge_grpo.py:195-197`

The paper claims $R_{\text{con}} = \mathbb{1}[v_{\text{swap}} = \text{flip}(v_{\text{orig}})]$, which requires comparing the model's output on the **original** prompt AND the **swapped** prompt for the same pair. But GRPO generates G completions for a **single** prompt per group — there is no mechanism to get the model's response to the *swapped version* of that same prompt.

Current workaround: a "proxy" that gives 0.5 if output is not "tie" and 0.0 if tie. This is **not consistency** — it's an "informativeness" bonus. It never actually checks position invariance.

**Proposed Solutions (ranked by feasibility)**:

| Solution | Pros | Cons |
|----------|------|------|
| **(A) Interleave original+swap in dataset** | Simple; each prompt appears twice; compute consistency as post-hoc cross-reference between adjacent batches | Requires careful data ordering; reward for one sample depends on output of another sample (violates independence within batch) |
| **(B) Two-pass consistency** | First pass: generate on all prompts. Second pass: generate on swapped versions. Compare offline. Use as offline reward signal. | Not online RL; becomes rejection sampling or offline RL |
| **(C) Proxy consistency (current)** | Trivial to implement | Not actually measuring consistency; scientifically dishonest if claimed as "consistency reward" |
| **(D) Paired sampling within group** | Use G/2 original + G/2 swapped within same group; pair them for consistency reward | Feasible! Each item in the dataset includes BOTH original and swapped prompts. For each group, generate G/2 from original, G/2 from swapped, compute consistency within the group. Requires custom sampler. |

**Recommendation**: Use **(D) Paired sampling**. Modify the dataset to include both versions per item. Within each GRPO group of G=8, generate 4 from original prompt and 4 from swapped prompt. For each (orig_i, swap_j) pair, compute consistency. This gives a valid consistency signal. Alternatively, accept **(C)** but rename it "informativeness reward" or "decisiveness reward" in the paper — and change the paper narrative to say you train for accuracy + decisiveness + calibration, not true position consistency. Then evaluate true position consistency at test time to show the reward transfers.

---

### CRITICAL-4: Calibration Reward Uses Fake Confidence

**File**: `scripts/train_judge_grpo.py:62-70`

```python
confidence = 0.8 if choice != "C" else 0.5
```

This is a **hardcoded heuristic**, not actual model confidence. The Brier score computed against this is meaningless:
- If model says A and gold is A: reward = 1 - (0.8 - 1.0)² = 0.96
- If model says A and gold is B: reward = 1 - (0.8 - 0.0)² = 0.36
- If model says C (tie): reward = 1 - (0.5 - 0/1)² = 0.75

This effectively degenerates to: correct → 0.96, wrong → 0.36, tie → 0.75. It's just a rescaled accuracy reward with a tie bonus. No actual calibration is being trained.

**Proposed Solutions**:

| Solution | Feasibility | Notes |
|----------|-------------|-------|
| **(A) Require model to output explicit confidence** | High | Change prompt template to demand `[[A, 0.85]]` format. Parse the confidence. Reward calibration based on parsed score. |
| **(B) Use token logprobs as confidence** | Medium | After generation, compute log P(choice token). Convert to probability. Use as confidence. Requires access to logits, which TRL does expose for GRPO. |
| **(C) Binary calibration (simplified)** | High | Define confidence = 1.0 for non-tie, 0.5 for tie. Reward = match(prediction, gold). This is just accuracy again. Not useful. |

**Recommendation**: Use **(A)**. Modify prompt template to ask for explicit confidence:
```
Output format: "[[A, confidence]]" or "[[B, confidence]]" or "[[C, confidence]]"
where confidence is a number between 0.5 and 1.0 indicating how sure you are.
```
Then parse both choice and confidence from the output. This gives a real calibration signal.

---

### MAJOR-1: RewardBench Dataset Split Name May Be Wrong

**File**: `scripts/prepare_data.py:87-88`

```python
ds = load_dataset("allenai/reward-bench", split="filtered")
```

According to the HuggingFace dataset page, RewardBench has splits: `raw` (5120 samples) and `filtered` (2990 samples). The code tries `"filtered"` first, then falls back to `"train"`. The fallback `"train"` does NOT exist — it will fail.

**Fix**: Change fallback to `"raw"`:
```python
try:
    ds = load_dataset("allenai/reward-bench", split="filtered")
except:
    ds = load_dataset("allenai/reward-bench", split="raw")
```

The `filtered` split (2990 samples) is correct for training since it has verified chosen-rejected rankings.

---

### MAJOR-2: RewardBench Field Names — `chosen`/`rejected` Are Full Text, Not Chat Messages

**File**: `scripts/prepare_data.py:55-57`

```python
question = item.get("prompt", item.get("instruction", ""))
chosen = item.get("chosen", "")
rejected = item.get("rejected", "")
```

According to the verified schema, RewardBench columns are: `prompt`, `chosen`, `rejected`, `chosen_model`, `rejected_model`, `subset`, `id`. The field names `prompt`, `chosen`, `rejected` are **correct**. However:

- `item.get("instruction", "")` fallback is unnecessary (field is always `prompt`)
- The `chosen`/`rejected` fields in RewardBench are plain text strings (full responses), which is what the code expects — this is correct.
- `item.get("id", "")` — the field is `id` (int64), not string. Should work but good to note.

**Verdict**: Field names are actually correct. The `instruction` fallback is harmless dead code.

---

### MAJOR-3: GRPOConfig Parameter Names May Be Wrong

**File**: `scripts/train_judge_grpo.py:163-177`

```python
training_config = GRPOConfig(
    num_train_epochs=3,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    learning_rate=args.lr,
    num_generations=args.group_size,
    max_completion_length=1024,
    max_prompt_length=2048,
    save_steps=100,
    ...
)
```

In TRL >=1.0, some parameter names changed:
- `num_generations` → correct (this is the group size G)
- `max_completion_length` → correct
- `max_prompt_length` → correct

However, the GRPOTrainer instantiation uses:
```python
GRPOTrainer(
    model=args.model_name,
    reward_funcs=reward_function,   # should be list or single callable
    args=training_config,
    train_dataset=dataset,
    peft_config=peft_config,
)
```

Issues:
- `reward_funcs` should be passed via `GRPOConfig(reward_funcs=...)` or as a parameter to `GRPOTrainer`. In TRL >=1.0, it's typically `reward_funcs` in GRPOConfig or `reward_funcs` param in GRPOTrainer constructor. Need to verify which.
- The dataset must have a `"prompt"` column. Current dataset has `"prompt"` and `"gold_label"` — `gold_label` will be passed via `**kwargs` to reward function. This should work.

**Fix**: Verify exact API. The safe approach for TRL 1.0+:
```python
trainer = GRPOTrainer(
    model=args.model_name,
    reward_funcs=[reward_function],
    args=training_config,
    train_dataset=dataset,
    peft_config=peft_config,
)
```

---

### MINOR-1: Train Data Path is Relative

**File**: `scripts/train_judge_grpo.py:119-120`

```python
parser.add_argument("--train_data", default="../data/train/judge_train.json")
parser.add_argument("--swap_data", default="../data/train/judge_swap.json")
```

Relative paths will break when running from different directories (e.g., `nohup` commands in EXPERIMENTS.md use absolute Python path but don't `cd` to scripts dir).

**Fix**: Use `PROJECT_ROOT` for default paths:
```python
parser.add_argument("--train_data", default=str(PROJECT_ROOT / "data/train/judge_train.json"))
parser.add_argument("--swap_data", default=str(PROJECT_ROOT / "data/train/judge_swap.json"))
```

---

### MINOR-2: Missing `evaluate_judge.py` and `evaluate_posthoc.py`

**File**: Referenced in `EXPERIMENTS.md:237-289` but not present in `scripts/`

The evaluation scripts don't exist yet. This isn't a bug per se (they're needed later), but should be noted for completeness.

---

### MINOR-3: Paper Claims ~3000 Instances but RewardBench Filtered Has 2990

The paper says "approximately 3,000 preference pairs" — accurate enough for RewardBench filtered (2990). After 70/15/15 split:
- Train: ~2093
- Val: ~449
- Test: ~448

This matches EXPERIMENTS.md ("~2100 train, ~450 test"). No issue here.

---

## Specific Modifications Needed for `train_judge_grpo.py`

### 1. Fix reward function signature and implementation

```python
def build_reward_function(reward_mode, alpha, beta, gamma):
    """Build a reward function compatible with TRL GRPOTrainer API."""
    
    def reward_function(completions, gold_label, **kwargs):
        """
        Args:
            completions: list[list[dict]] — each item is [{"role": "assistant", "content": "..."}]
            gold_label: list[str] — gold labels from dataset column
        Returns:
            list[float] — reward for each completion
        """
        rewards = []
        for i, completion in enumerate(completions):
            text = completion[0]["content"]
            gold = gold_label[i]
            
            parsed = parse_judge_output(text)
            
            # Accuracy
            acc = 1.0 if parsed["choice"] == gold else 0.0
            
            if reward_mode == "accuracy":
                r = alpha * acc
            elif reward_mode == "acc_consist":
                # Proxy: penalize ties (not true consistency, but trainable signal)
                decisiveness = 0.0 if parsed["choice"] == "C" else 0.5
                r = alpha * acc + beta * decisiveness
            elif reward_mode == "acc_calib":
                conf = parsed["confidence"]
                correct = 1.0 if parsed["choice"] == gold else 0.0
                brier = (conf - correct) ** 2
                calib = 1.0 - brier
                r = alpha * acc + gamma * calib
            else:  # full
                decisiveness = 0.0 if parsed["choice"] == "C" else 0.5
                conf = parsed["confidence"]
                correct = 1.0 if parsed["choice"] == gold else 0.0
                brier = (conf - correct) ** 2
                calib = 1.0 - brier
                r = alpha * acc + beta * decisiveness + gamma * calib
            
            rewards.append(r)
        return rewards
    
    return reward_function
```

### 2. Fix parse_judge_output for confidence extraction

Change prompt template to request explicit confidence, then parse:
```python
def parse_judge_output(text):
    """Parse judge output: [[A, 0.85]] or [[A]] format."""
    # Try format with confidence: [[A, 0.85]]
    match = re.search(r'\[\[(A|B|C),?\s*([\d.]+)?\]\]', text)
    if match:
        choice = match.group(1)
        confidence = float(match.group(2)) if match.group(2) else (0.8 if choice != "C" else 0.5)
        confidence = max(0.5, min(1.0, confidence))  # clamp
    else:
        choice = "C"  # unparseable → tie
        confidence = 0.5
    return {"choice": choice, "confidence": confidence}
```

### 3. Uncomment and fix trainer instantiation

```python
reward_fn = build_reward_function(args.reward_mode, args.alpha, args.beta, args.gamma)

trainer = GRPOTrainer(
    model=args.model_name,
    reward_funcs=[reward_fn],
    args=training_config,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(str(output_dir / "final_model"))
```

### 4. Fix default data paths to use absolute paths

```python
parser.add_argument("--train_data", default=str(PROJECT_ROOT / "data/train/judge_train.json"))
parser.add_argument("--swap_data", default=str(PROJECT_ROOT / "data/train/judge_swap.json"))
```

---

## Supplements to EXPERIMENTS.md

### Missing Steps

1. **EXP-004/005 gap**: Numbering jumps from EXP-003 (pilot) to EXP-006. Reserve EXP-004 for "base model evaluation" and EXP-005 for "prompt template validation".

2. **Prompt template with confidence**: The prompt template in `prepare_data.py` asks for `[[A]]/[[B]]/[[C]]` but does NOT ask for confidence. If using calibration reward with explicit confidence, the template needs updating:
   ```
   Output format: "[[X, confidence]]" where X is A, B, or C, and confidence is between 0.5 and 1.0
   ```

3. **Evaluation swap data**: `EXPERIMENTS.md` references `data/eval/rewardbench_test_swap.json` (line 256) but `prepare_data.py` only generates `rewardbench_test.json` for eval. Need to also generate swapped eval data.

4. **Base model accuracy check**: Before training, verify base Qwen2.5-7B-Instruct accuracy on RewardBench is 60-65% as claimed. If it's <30% or >70%, GRPO sweet spot may not apply.

5. **Format compliance rate**: Add a pre-training check — run base model on 50 prompts, measure what % produce parseable `[[A/B/C]]` output. If <50%, the format reward penalty (reward=0 for unparseable) will dominate training signal.

---

## Summary of Required Actions

| Priority | Action | File |
|----------|--------|------|
| CRITICAL | Fix reward function API signature | `train_judge_grpo.py` |
| CRITICAL | Uncomment trainer code | `train_judge_grpo.py` |
| CRITICAL | Decide on consistency reward strategy (proxy vs paired) | `train_judge_grpo.py` + paper |
| CRITICAL | Fix calibration to use real confidence | `train_judge_grpo.py` + `prepare_data.py` |
| MAJOR | Fix fallback split name in data loading | `prepare_data.py` |
| MAJOR | Fix default data paths (relative → absolute) | `train_judge_grpo.py` |
| MAJOR | Verify GRPOTrainer constructor API | `train_judge_grpo.py` |
| MINOR | Generate swapped eval data | `prepare_data.py` |
| MINOR | Write evaluation scripts | New files needed |
| MINOR | Add base model accuracy pre-check | `EXPERIMENTS.md` |
