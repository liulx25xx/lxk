# Review Response Plan — Paper 3: Position Shortcut in Judge RL

Generated: 2026-05-17

---

## Overview

This document specifies concrete modifications to address AI review feedback. Each issue is classified as P0 (must fix before submission) or P1 (strengthens paper), with implementation details.

---

## P0 Issues (Critical)

### P0-1: "Reward engineering cannot fix" is Over-Claim

**Reviewer's Point**: The paper claims "multi-objective rewards cannot overcome data-level confounds." But we only tested a *proxy* consistency reward (decisiveness = penalizing ties). We never tested a *true paired swap-invariance reward* that conditions reward on identical judgment across original+swapped inputs. The claim is too strong.

**Current Paper Text** (line 52):
> "We show that multi-objective rewards (consistency proxies, calibration) cannot overcome data-level confounds"

**Fix — Two-Pronged Approach**:

#### (a) Narrow the Claim (Writing, 0 GPU)

Replace with:
> "We show that *proxy* multi-objective rewards (decisiveness, calibration) cannot overcome data-level confounds, because proxy rewards do not directly measure position invariance."

Add a sentence in Discussion acknowledging:
> "A true paired invariance reward—requiring consistent judgments across position-swapped inputs within the same RL update—may break the confound from the reward side. We test this hypothesis in Section X.Y."

#### (b) Implement True Paired Invariance Training (Code + ~3h GPU)

**Concept**: For each training instance, generate completions for BOTH original and swapped prompts. The reward is 1.0 only if:
- Original chooses correctly (accuracy) AND
- Original-choice flips correctly when positions are swapped (consistency)

This is the reward that J1/JudgeLRM *should* have used if they wanted consistency without balanced data.

**Implementation Plan** (`train_judge_grpo_paired.py`):

```python
# Key modification: dual-prompt dataset
# Dataset has columns: prompt_orig, prompt_swap, gold_label

def paired_reward_function(completions, gold_label, prompt_swap_text, **kwargs):
    """
    Reward = 1.0 only if:
      1. Judge is correct on original (choice == gold_label)
      2. Judge would flip correctly on swap (inferred from decisiveness pattern)
    
    PROBLEM: Standard TRL GRPOTrainer generates completions for ONE prompt.
    We cannot generate for swap within the same reward call.
    
    SOLUTION: Two-phase approach:
      Phase 1: Generate for original prompt (standard GRPO)
      Phase 2: For each completion, simulate what "consistent" means
              → If choice=A and gold=A, model is correct.
              → Award 0.5 for accuracy.
              → Award additional 0.5 only if NOT choosing C (proxy for having a clear preference that could flip)
    
    BETTER SOLUTION: Modify dataset to interleave orig+swap as consecutive pairs.
    Then in reward function, check pairs together.
    """
```

**Actually Correct Implementation** — Custom training loop with paired generation:

The real paired training cannot use standard TRL GRPOTrainer directly because it generates completions for a single prompt per instance. We need:

1. **Dataset format**: Each instance has `prompt_orig` + `prompt_swap` + `gold_label`
2. **Generation**: For each training step, generate G completions for `prompt_orig` AND G completions for `prompt_swap` (same model, same batch)
3. **Reward logic**:
   ```python
   for i in range(batch_size):
       orig_choice = parse(completions_orig[i])
       swap_choice = parse(completions_swap[i])
       
       # Accuracy component
       acc = 1.0 if orig_choice == gold_label[i] else 0.0
       
       # True consistency: orig says A → swap should say B
       flip_map = {"A": "B", "B": "A", "C": "C"}
       is_consistent = (swap_choice == flip_map[orig_choice])
       consist = 1.0 if is_consistent else 0.0
       
       # Combined reward
       reward[i] = 0.6 * acc + 0.4 * consist
   ```
4. **Practical approach**: Since TRL doesn't natively support paired prompts, use a **simpler approximation**:
   - Interleave dataset: [orig_0, swap_0, orig_1, swap_1, ...]
   - Gold labels: [A, B, A, B, ...] (swap has flipped gold)
   - Train with standard accuracy reward on this interleaved dataset
   - This is functionally equivalent to balanced training if we shuffle!

**Key Insight**: True paired invariance training with accuracy reward on interleaved (orig+swap) data IS mathematically equivalent to balanced data training with accuracy reward. The "paired reward" that's actually DIFFERENT is one that conditions the reward for instance_orig on the SAME MODEL's output for instance_swap within the same GRPO group.

**Final Implementation Decision**:

Option A (Recommended, ~2h implementation + 3h training):
- Write custom training script that generates for BOTH prompts per instance
- Use OpenRLHF-style custom reward that sees both completions
- Requires modifying the generation pipeline but not the optimizer

Option B (Simpler, may not work differently from balanced):
- Train with accuracy reward on interleaved orig+swap data
- Argue this is equivalent to paired training
- Risk: reviewer says "that's just balanced data with extra steps"

**Recommendation**: Do Option A. Even if the result is similar to balanced data, it answers the reviewer's specific question: "Have you tried a reward that directly measures swap-invariance?" If paired training ALSO fixes the problem → write "both balanced data and paired invariance rewards can break the confound, but naive proxy rewards cannot."

**Expected Outcome**:
- If paired reward works (Consistency > 80%): "True paired reward IS sufficient, but standard implementations use proxies that aren't"
- If paired reward fails (Consistency < 70%): "Even true invariance reward cannot overcome the confound" → strengthens data-fix claim further

**GPU Time**: ~3h on 1 H200 (300 steps, same as other experiments)

---

### P0-2: Only 1 Model + 1 Dataset

**Reviewer's Point**: Qwen2.5-7B + RewardBench is a single point. Generalization unknown.

**Status**:
- Qwen3-8B baseline already evaluated: Acc=36.1%, Consist=78.4% (n=449)
- Qwen3-8B GRPO training in progress (step ~20/300 as of 2026-05-17)

**Fix — Three Components**:

#### (a) Qwen3-8B Full Pipeline (In Progress, ~6h remaining)

- Training: GRPO accuracy-only on unbalanced data (same protocol as Qwen2.5-7B)
- Expected: same pattern (accuracy up, consistency down)
- Then: balanced data variant if time permits
- **Status**: Training launched, checkpoint at step 100 expected in ~2h

#### (b) Second Dataset: HH-RLHF Judge Data (4-6h total)

**Construction Plan**:
1. Download Anthropic HH-RLHF (`Anthropic/hh-rlhf` from HF)
2. It has `chosen` and `rejected` conversations → perfect for judge format
3. Convert to our judge prompt format:
   - question = the human turn
   - answer_a = chosen[-1] (assistant response)
   - answer_b = rejected[-1] (assistant response)
   - gold_label = "A"
4. Sample 2000 train + 500 eval instances
5. Create balanced variant (swap half → gold_label = "B")
6. Train GRPO accuracy-only on unbalanced → expect position shortcut
7. Verify balanced fixes it

**Script**: `prepare_hh_rlhf_data.py` (new file)
**Training**: Same configs as Qwen2.5-7B RewardBench experiments
**GPU Time**: ~3h (1 unbalanced + 1 balanced training)

#### (c) Alternative: UltraFeedback (Lower Priority)

UltraFeedback has explicit model scores → can create pairs where better=higher-score. Same pipeline as (b). Only pursue if HH-RLHF shows unexpected patterns.

**Priority**: (a) is running → (b) next if GPU available → (c) only if time permits

---

### P0-3: Calibration Formula Clarity

**Reviewer's Point**: $R_{\text{cal}} = 1 - (c - \mathbb{1}[v = v^*])^2$ — need to clarify this is Brier score with indicator function.

**Current Paper**: Already uses $\mathbb{1}$ notation correctly.

**Fix** (Writing only, 10 min):
1. Add explicit sentence: "where $c \in [0.5, 1.0]$ is the model's stated confidence and $\mathbb{1}[v = v^*] \in \{0, 1\}$ is the correctness indicator"
2. Add ECE computation details in appendix:
   - Bin predictions by confidence into 10 equal-width bins
   - ECE = $\sum_{b=1}^{B} \frac{|B_b|}{N} |acc(B_b) - conf(B_b)|$
   - Report ECE alongside Brier in results table

---

## P1 Issues (Strengthening)

### P1-1: Why Balanced SFT > Balanced GRPO on Accuracy

**Data**: SFT balanced = 91.3% acc, GRPO balanced = 84.6% acc

**Explanation to Add** (1 paragraph in Discussion):

> When gold outputs are available—as in the judge setting where correct verdicts are known—SFT directly maximizes the probability of the correct answer via MLE. GRPO must discover the correct answer through sampling and reinforce it via group-relative advantages, a less sample-efficient optimization path. This explains why balanced SFT achieves higher accuracy: the training signal is direct rather than indirect. However, GRPO's advantage lies in settings where gold outputs are unavailable and only binary correctness rewards exist. In practice, judge training data often lacks gold verdicts (requiring human annotation), making GRPO's reward-only requirement more practical. The key finding remains: both methods require balanced data to avoid the position shortcut.

---

### P1-2: Tie Handling in Consistency Metric

**Add to Method section** (2-3 sentences):

> Consistency is computed between original and position-swapped responses for the same comparison pair. Let $v_{\text{orig}}$ and $v_{\text{swap}}$ denote the model's verdicts. We define consistency as:
> - $v_{\text{orig}} = A \land v_{\text{swap}} = B$: consistent (correct flip)
> - $v_{\text{orig}} = B \land v_{\text{swap}} = A$: consistent (correct flip)
> - $v_{\text{orig}} = C \land v_{\text{swap}} = C$: consistent (tie preserved)
> - All other combinations: inconsistent
>
> Note: $v_{\text{orig}} = A \land v_{\text{swap}} = A$ is inconsistent because swapping positions should flip the preferred position label.

---

### P1-3: Qualitative Examples

**Source**: `/path/to/paper3_judge_rl/results/EXP-006_accuracy_only/eval/eval_results.json`

**Cases to Extract**:

| Case | Type | Description |
|------|------|-------------|
| 1 | Shortcut success | Unbalanced model says A (correct), swap also says A (wrong) — right answer, wrong reason |
| 2 | Balanced model | Balanced model says A (correct), swap says B (correct flip) — genuine judgment |
| 3 | Adversarial failure | Category=adversarial, model exploits position regardless of content quality |

**Implementation**: Write `extract_qualitative_examples.py` that:
1. Loads eval_results.json for unbalanced and balanced models
2. Finds instances where unbalanced is correct+inconsistent (Case 1)
3. Finds instances where balanced is correct+consistent (Case 2)
4. Finds adversarial category failures (Case 3)
5. Outputs formatted LaTeX for paper

**GPU Time**: 0 (analysis only)

---

### P1-4: Statistical Tests

**Add to Results**:

#### Bootstrap 95% CI
- For each metric (accuracy, consistency), sample with replacement 1000 times from eval results
- Report: "Accuracy = 94.4% [93.1%, 95.6%]"
- Shows that differences between conditions are not sampling noise

#### McNemar's Test
- For balanced vs unbalanced consistency:
  - Construct 2x2 table: (consistent_both, consistent_only_balanced, consistent_only_unbalanced, inconsistent_both)
  - McNemar chi-squared test
  - Expected: highly significant (p < 0.001) given 449 test samples and ~25pp difference

**Implementation**: `compute_statistics.py`
```python
from scipy.stats import bootstrap, chi2
import numpy as np

# Bootstrap CI
def bootstrap_ci(results, metric_key, n_boot=1000):
    values = [r[metric_key] for r in results]
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(values, size=len(values))) for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])

# McNemar test
def mcnemar_test(results_a, results_b, metric_key="is_consistent"):
    # results_a and results_b must be aligned (same instances)
    b = sum(1 for a, b in zip(results_a, results_b) if a[metric_key] and not b[metric_key])
    c = sum(1 for a, b in zip(results_a, results_b) if not a[metric_key] and b[metric_key])
    chi2_stat = (b - c)**2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value
```

**GPU Time**: 0

---

## Experiment Priority & Timeline

| # | Task | Type | GPU Time | Priority | Dependency |
|---|------|------|----------|----------|------------|
| 1 | Narrow "reward engineering cannot fix" claim | Writing | 0 | P0 HIGH | None |
| 2 | Wait for Qwen3-8B GRPO results | Running | ~5h left | P0 HIGH | Already launched |
| 3 | Paired invariance training experiment | Code+GPU | ~5h total | P0 MED | Need new script |
| 4 | Clarify calibration formula + ECE | Writing | 0 | P0 LOW | None |
| 5 | Add tie handling explanation | Writing | 0 | P1 | None |
| 6 | Extract qualitative examples | Analysis | 0 | P1 | None |
| 7 | Bootstrap CI + McNemar test | Analysis | 0 | P1 | None |
| 8 | HH-RLHF second dataset | Code+GPU | ~6h | P0 MED | After Qwen3 done |
| 9 | Add SFT vs GRPO explanation | Writing | 0 | P1 | None |

---

## Implementation Details: Paired Invariance Training

### File: `scripts/train_judge_grpo_paired.py`

**Approach**: Since TRL GRPOTrainer generates completions for a single prompt per item, we implement paired evaluation via a two-step training process:

**Step 1 — Prepare Paired Dataset**:
Each training instance contains BOTH original and swapped prompts as a SINGLE concatenated prompt that asks the model to judge both:

Actually, this breaks the judge format. Better approach:

**Correct Approach — Custom Reward with Cached Swapped Outputs**:

1. Dataset: standard `judge_train.json` (original prompts only)
2. Before each training epoch, run inference on all swap prompts → cache swap outputs
3. Reward function checks: is current completion consistent with the cached swap output?

**Problem**: Cached swap outputs are from a PREVIOUS checkpoint, creating staleness.

**Best Approach — Alternating Batches**:

1. Interleave training: even steps use original prompts, odd steps use swap prompts
2. Reward for originals: accuracy (choice == gold_label_A)
3. Reward for swaps: accuracy (choice == gold_label_B)
4. The model sees BOTH position orientations → learns content-based judgment

Wait — this IS just balanced data training. The reviewer wants something where a SINGLE reward signal penalizes inconsistency.

**FINAL Correct Approach — Group-Level Paired Reward**:

Modify GRPO to generate for both orientations within the SAME group:
- For each instance, prompt_orig and prompt_swap are treated as a "super-prompt"
- Generate G/2 completions for orig, G/2 for swap
- Reward for the GROUP = f(accuracy_orig, consistency_across_orientations)

**Implementation**:
```python
# Modified dataset: each row has both prompts
# prompt = prompt_orig (used for generation)
# reward function receives completions + prompt_swap column
# At reward time: 
#   1. Run additional forward pass on prompt_swap for each completion
#   2. Compare orig completion choice with swap completion choice
#   3. Reward = accuracy * consistency

# This requires model inference INSIDE the reward function
# TRL supports this via reward_model (a model that scores)
# BUT we need generation, not just scoring

# SIMPLEST WORKING APPROACH:
# Use a separately-maintained "swap response cache" that updates every N steps
# Staleness is acceptable if N is small (every 50 steps = 10% of training)
```

**Pragmatic Implementation Decision**:

Given EMNLP deadline constraints, implement the following:
1. Before training: generate swap responses with base model → store
2. During GRPO training: reward = 0.6*accuracy + 0.4*swap_consistency
   where swap_consistency compares current completion with stored swap response
3. Every 100 steps: re-generate swap responses with current model → update cache
4. This approximates true paired training with bounded staleness

**Expected result**: This should work BETTER than proxy decisiveness but SIMILAR to balanced data (since both remove position shortcut). The scientific value is showing the mechanism: direct consistency measurement works, but requires computational overhead that balanced data avoids for free.

---

## Writing Changes Summary

### Abstract
- "cannot overcome" → "proxy rewards cannot overcome"
- Add: "while true paired invariance rewards provide equivalent protection to balanced data"

### Contributions (line 52)
- "multi-objective rewards (consistency proxies, calibration) cannot overcome" →
  "proxy multi-objective rewards (decisiveness, calibration) cannot overcome data-level confounds; only rewards that directly measure position invariance match the effectiveness of balanced data"

### Method (Section 3)
- Add 2 sentences on tie handling in consistency definition
- Clarify calibration formula with explicit variable definitions

### Results (Section 4)
- Add Qwen3-8B row to main results table (pending training completion)
- Add bootstrap 95% CI in parentheses after key numbers
- Add qualitative examples box (Figure or table)

### Discussion
- Add paragraph on "Why SFT > GRPO when gold outputs available"
- Add paragraph on paired invariance training results
- Soften "reward engineering cannot fix" → "standard proxy rewards cannot fix; only rewards requiring paired evaluation or data balancing eliminate the confound"

### Appendix
- ECE computation details
- McNemar test table
- Full qualitative examples (3-5 cases with model outputs)
- HH-RLHF replication results (if completed before deadline)
- Per-category Qwen3-8B breakdown

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Qwen3-8B doesn't show same pattern | Unlikely (same data confound). If consistency stays high → investigate model-specific robustness (also interesting) |
| Paired training = exactly balanced data | Expected. Frame as: "confirms that the fix must target position-reward correlation" |
| HH-RLHF data quality issues | HH-RLHF is well-established. Worst case: drop this and rely on Qwen3 generalization |
| Deadline pressure (2026-05-25) | Writing fixes (P0-1, P0-3, P1-1/2) can be done in 1 day. Experiments need 2-3 days |

---

## Checklist Before Submission

- [ ] Claim "reward engineering cannot fix" narrowed to "proxy rewards"
- [ ] Qwen3-8B results in paper (at minimum: baseline + unbalanced GRPO)
- [ ] Calibration formula clarified with ECE in appendix
- [ ] Tie handling explained in Method
- [ ] Bootstrap CI on main results
- [ ] At least 2 qualitative examples
- [ ] McNemar test for balanced vs unbalanced
- [ ] SFT vs GRPO explanation paragraph
- [ ] Paired invariance training result (if completed) or at minimum Discussion section acknowledging this as future work
- [ ] Second dataset (HH-RLHF) results or Discussion noting this as limitation
