# Deep Insight Analysis: Paper 3 — Self-Consistent Calibrated Judges via RL

**Date**: 2026-05-17  
**Purpose**: Elevate Paper 3 from "engineering contribution" (composite reward = obvious recipe) to genuinely surprising scientific finding.

---

## 1. The Problem with the Current Framing

The current paper says: "We combine accuracy + consistency + calibration rewards to train better judges."

**Why this is engineering, not insight**: Any reasonable reader would predict that training with 3 objectives beats training with 1. It's obvious. The abstract could be written *before* running experiments. This violates the user's research taste criterion: if the outcome is predictable, it's not a paper.

---

## 2. Genuinely Surprising Insights (Ranked by Non-Obviousness)

### Insight A: Consistency Training Is Implicit Debiasing That *Improves* Accuracy [⭐ RECOMMENDED]

**The claim**: Training a judge to ignore position (via consistency reward) actually improves its *accuracy* — not just its consistency. The accuracy-only model achieves *lower* final accuracy than the model trained with accuracy + consistency.

**Why this is surprising**: The naive expectation is a tradeoff — removing positional shortcuts should *reduce* accuracy because the model loses an easy cue (in imbalanced datasets, position correlates weakly with gold labels). Instead, consistency training forces the model to develop genuine content-comparison capabilities rather than surface heuristics. The "shortcut" was never contributing real judgment quality; removing it exposes the model to the actual signal.

**Analogy**: This is similar to how dropout *improves* generalization — removing something from the model paradoxically makes it stronger. Or how data augmentation (which adds noise/invariances) can improve test accuracy. The shortcut was stealing gradient from the real signal.

**Empirical signature to verify**:
- `Acc(Acc+Con) > Acc(Acc-only)` — the accuracy-trained model should have *lower* test accuracy than the consistency-augmented model
- Especially visible in **adversarial categories** (llmbar-adver-*) where position is maximally misleading
- Baseline data already shows: `llmbar-adver-GPTInst: acc=0.40, con=0.80` — the model is consistent but *wrong*, meaning it has learned a stable-but-wrong heuristic. RL with consistency reward should break this stable-wrong state.

**Why post-hoc swap doesn't achieve this**: Post-hoc swap corrects the *output* but doesn't change the *model*. The internal representations remain biased. It's an ensemble trick that masks the problem rather than fixing it. A model trained with consistency has different internal representations — you can prove this with probing.

---

### Insight B: Accuracy-Only RL Creates "Confident Errors" — Calibration Reward Prevents Reward Hacking

**The claim**: Training a judge with accuracy-only reward actively *worsens* its calibration — the model becomes overconfident on everything including its mistakes. Adding calibration reward doesn't just fix calibration; it prevents the model from developing degenerate confidence patterns that indicate reward hacking.

**Why this is surprising**: RL for accuracy should make the model better at judging. Why would it make confidence worse? Because GRPO reinforces high-reward completions, and the model learns that assertive outputs ("clearly, A is better" → [[A, 0.95]]) receive higher reward than hedged outputs. The model is hacking the format, not improving judgment. Calibration reward acts as a *regularizer against reward hacking* — it prevents the model from exploiting surface patterns that correlate with reward without corresponding to genuine improvement.

**Safety/alignment angle**: This connects to a broader problem in RLHF — reward hacking through confident presentation. A judge trained only for accuracy may become a confident but unreliable oracle, which is worse than a slightly less accurate but honest judge when used downstream for reward modeling.

**Empirical signature**:
- `ECE(Acc-only) > ECE(Base)` — accuracy-RL makes calibration *worse* than no training
- `ECE(Full) < ECE(Base)` — only the full composite recovers calibration
- Qualitative: accuracy-only model outputs confidence ≈1.0 on everything; full model outputs varied confidence

---

### Insight C: Training-Time Optimization Has Strictly Higher Ceiling Than Post-Hoc Correction

**The claim**: There is a theoretical and empirical ceiling to post-hoc methods that training-time optimization breaks through. Post-hoc swap can never exceed the accuracy of the *better* orientation (it averages), while training can make *both* orientations correct.

**Why this is surprising (partially)**: It's partially obvious that training changes the model while post-hoc doesn't. The non-obvious part is *quantifying the gap* and showing that it's large enough to matter. Specifically:
- Post-hoc swap accuracy = (acc_orig + acc_swap) / 2 (bounded by average of two runs)
- RL-trained accuracy can exceed BOTH original and swapped accuracy of the base model

**Empirical requirement**: Show that trained model accuracy > max(base_orig_accuracy, base_swap_accuracy, posthoc_average_accuracy). This proves training discovered strategies that neither orientation of the base model had.

---

### Insight D: Domain-Specific Reward Interactions (Interesting but Incremental)

Different RewardBench domains respond differently to each reward component:
- **Adversarial categories** (llmbar-adver-*): consistency reward helps most (position is maximally exploited)
- **Math/Code**: accuracy reward dominates (clear right/wrong, less position bias)  
- **Safety/Chat**: calibration matters most (uncertainty about correctness should propagate)

This is interesting for practitioners but not surprising enough to be the core insight.

---

## 3. Recommended Core Insight and Research Question

### Core Insight (1 sentence):

**"Consistency training is implicit debiasing: forcing a judge to produce position-invariant verdicts removes shallow shortcuts and improves the underlying judgment quality, yielding *higher accuracy* than training for accuracy alone."**

### Research Question (1 sentence):

**"Does training for position invariance improve judge accuracy by preventing reliance on positional shortcuts, and does this effect compound with calibration training to produce judges that are simultaneously more accurate, more consistent, and better calibrated than any single-objective training?"**

### Why This RQ is Non-Obvious:

The naive answer to "does adding a consistency constraint help accuracy?" is "no, it should hurt — you're removing information (position cues)." The surprising finding is that position cues were *noise*, not signal, and removing them is *debiasing* that exposes the model to real content differences. This is the same mechanism as invariance-based self-supervised learning (where removing shortcut features improves downstream task performance).

---

## 4. Redesigned Experiment Plan for Insight Verification

Current experiments (4 ablations) remain necessary as infrastructure, but the *analysis* and *additional experiments* change:

### Experiment 1: Position Shortcut Detection (CRITICAL)

**Goal**: Prove that accuracy-only RL learns position shortcuts that consistency training prevents.

| Analysis | Method | Expected Finding |
|----------|--------|-----------------|
| Position preference ratio | For Acc-only model, compute P(choose A) vs P(choose B). Compare with Full model. | Acc-only: skewed (e.g., 60/40). Full: balanced (50/50). |
| Accuracy on position-shuffled vs fixed test | Eval Acc-only on test set where gold is always in position A vs always in B | Acc-only: asymmetric accuracy. Full: symmetric. |
| Per-category accuracy gain from +Con | Compare Acc vs Acc+Con per RewardBench category | Largest gains in adversarial categories where position correlates with distractor quality |

### Experiment 2: Accuracy Gain from Consistency (THE KEY RESULT)

**Goal**: Show `Accuracy(Acc+Con) ≥ Accuracy(Acc-only)` — consistency doesn't trade off against accuracy, it *helps* accuracy.

| Metric | Acc-only | Acc+Con | Expected |
|--------|----------|---------|----------|
| Overall accuracy | X% | Y% (Y ≥ X) | Consistency helps or is neutral |
| Adversarial accuracy | Low | Higher | Biggest gain where position was exploited |
| Consistency | Low | High | By construction |

**If Y < X**: The insight needs revision — consistency *does* trade off. But this would also be interesting (as a negative result): "when does removing shortcuts hurt?" Still publishable but different story.

### Experiment 3: Calibration Prevents Reward Hacking

**Goal**: Show accuracy-only RL worsens calibration, and calibration reward prevents overconfident reward hacking.

| Metric | Base | Acc-only | Full |
|--------|------|----------|------|
| Mean confidence | ~0.8 (uniform) | ~0.95 (inflated) | ~0.75 (varied, calibrated) |
| ECE | Baseline | Worse | Best |
| Confidence on WRONG answers | 0.8 | 0.9+ | 0.5-0.6 |

**Key diagnostic**: Plot confidence vs correctness for each model. Acc-only should show a "wall of confidence" at 0.95 regardless of correctness. Full should show a clean monotonic calibration curve.

### Experiment 4: Probing Analysis (If Time Permits — Strong Differentiator)

**Goal**: Show that consistency training changes internal representations, not just output distributions.

- Train linear probes on hidden states to predict "position A was chosen" (binary)
- **Expected**: Acc-only model has high probe accuracy (position is encoded). Full model has low probe accuracy (position information is suppressed).
- This proves the training is doing *representation-level debiasing*, not just output-level correction.

### Experiment 5: Post-Hoc Ceiling Comparison

**Goal**: Quantify the gap between training-time and inference-time correction.

| Method | Accuracy | Consistency | ECE | Inference Cost |
|--------|----------|-------------|-----|----------------|
| Base | 80% | 83% | 0.14 | 1x |
| Post-hoc Swap (2x) | ~83% | ~95% | 0.14 | 2x |
| Full RL (Ours) | TBD | TBD | TBD | 1x |
| Post-hoc Swap on Full RL | TBD | TBD | TBD | 2x |

**Key claim**: Full RL at 1x cost matches or beats Post-hoc at 2x cost. AND post-hoc on top of Full RL gives diminishing returns (training already internalized the invariance).

---

## 5. How This Changes the Paper Framing

### OLD Framing (Engineering):
> "We combine 3 rewards and show full > subset. First to do composite RL for judges."

### NEW Framing (Insight-Driven):

> "We discover that training judges for position invariance functions as implicit debiasing that improves accuracy — not despite constraining the model, but *because* the constraint removes harmful shortcuts. This reveals that the positional patterns exploited by accuracy-only RL are noise masquerading as signal. Our composite training produces judges that are more accurate, consistent, and calibrated than any single-objective approach, with the key mechanism being shortcut suppression rather than simple multi-task learning."

### Revised Title Options:
1. "Shortcut Suppression Improves Judgment: How Consistency Training Makes Judges More Accurate"
2. "Beyond Post-Hoc: Training Inherently Reliable LLM Judges via Invariance-Driven RL"
3. "Consistency as Debiasing: Why Position-Invariant Training Improves Judge Accuracy"

### Revised Abstract Structure:
1. **Problem**: LLM judges have position bias. Current fixes are post-hoc.
2. **Surprising finding**: Training for consistency (position invariance) unexpectedly improves *accuracy* by suppressing positional shortcuts.
3. **Mechanism**: Positional cues are noise that accuracy-only RL exploits. Consistency reward acts as implicit debiasing, forcing content-based evaluation.
4. **Evidence**: Controlled ablation across 5 configurations + per-category analysis + representation probing.
5. **Practical implication**: Single-pass trained judge matches post-hoc correction at half the cost.

---

## 6. Risk Assessment

| Scenario | Probability | Impact | Mitigation |
|----------|-------------|--------|------------|
| Acc+Con accuracy ≈ Acc-only (no gain) | 30% | High — core insight weakened | Pivot to "consistency comes free" (no accuracy cost), still interesting but weaker |
| Acc+Con accuracy < Acc-only (tradeoff) | 20% | High — must change story entirely | Pivot to "When Does Invariance Training Help vs Hurt?" — characterize conditions |
| Acc-only doesn't worsen calibration | 25% | Medium — Insight B weakened | Emphasize Insight A instead; calibration becomes secondary |
| Probing shows no position encoding difference | 30% | Low — probing is optional extra | Drop probing, rely on behavioral evidence |
| Post-hoc swap beats Full RL on accuracy | 15% | Medium — practical value claim weakened | Emphasize that Full RL matches post-hoc on consistency while being 2x cheaper, even if post-hoc has slight accuracy edge |

**Critical experiment to watch**: The Acc-only vs Acc+Con accuracy comparison. This single result determines which story the paper tells.

---

## 7. Connection to Literature (Theoretical Grounding)

The insight connects to established ML principles:
1. **Invariance-based learning** (contrastive learning, data augmentation): Adding invariances to training improves generalization because it removes shortcut solutions. Our consistency reward is doing exactly this for judges.
2. **Shortcut learning** (Geirhos et al., 2020): Neural networks prefer shortcuts (texture over shape, position over content). RL with accuracy-only reinforces shortcuts. Our multi-objective training is a form of "shortcut suppression."
3. **Regularization through constraints**: Adding constraints (consistency, calibration) to the optimization landscape pushes the model away from degenerate solutions toward ones with better inductive biases.

This theoretical grounding elevates the paper from "we tried 3 objectives and it worked" to "we show that invariance-driven training prevents shortcut learning in judges, connecting to established principles of robust representation learning."

---

## 8. Summary of Recommendations

| Aspect | Recommendation |
|--------|---------------|
| **Core insight** | Consistency training is implicit debiasing that improves accuracy by suppressing positional shortcuts |
| **Research question** | Does position-invariance training improve judge accuracy (not just consistency)? |
| **Key experiment** | Acc-only vs Acc+Con accuracy comparison, especially on adversarial categories |
| **Additional experiments** | Position preference analysis, calibration degradation analysis, probing (optional) |
| **Paper framing** | From "first multi-objective RL for judges" → "invariance training improves judgment quality through shortcut suppression" |
| **Theoretical connection** | Shortcut learning literature + invariance-based representation learning |
| **Fallback if insight fails** | "Consistency comes free" (no tradeoff) or "When does invariance help/hurt?" |

---

## 9. Immediate Action Items

1. **Wait for EXP-006 vs EXP-007a/007b results** — the accuracy comparison is THE key data point
2. **Add per-category analysis** to eval script — need category-level breakdown to show where consistency helps most
3. **Add position preference analysis** — for each model, compute P(output A) vs P(output B) to quantify positional bias
4. **Add confidence distribution analysis** — histogram of output confidence for each model to show Insight B
5. **Prepare probing experiment** (optional) — extract hidden states, train linear probe for position prediction
6. **Once results arrive**: If Acc+Con ≥ Acc on accuracy → double down on Insight A. If tradeoff → pivot to characterization paper.

---

## 10. NEW Experiments (Added 2026-05-17 — Anti-Gaming Framing)

These experiments support the updated paper narrative: "Accuracy-only RL teaches judges to game, not to judge. Multi-objective reward acts as anti-gaming regularization."

### EXP-NEW-1: Reward Hacking Detection

**Goal**: Directly measure whether accuracy-only RL (EXP-006) produces shortcut-exploiting behavior, and whether full composite RL (EXP-009) avoids it.

**Protocol**:
For EXP-006 (accuracy-only) and EXP-009 (full composite) trained models, compute:

| Metric | Method | What it shows |
|--------|--------|---------------|
| **Position preference** | P(choose A) vs P(choose B) across full test set | Acc-only should skew (e.g., 60/40 toward position A). Full should be balanced (~50/50). |
| **Length preference** | Correlation between verdict and response length difference | Acc-only should prefer longer responses. Full should show no correlation. |
| **Format preference** | Accuracy on pairs where the "worse" response has better formatting | Acc-only should systematically pick the well-formatted response. Full should resist. |
| **Phantom accuracy rate** | Percentage of correct verdicts where rationale references only surface features (position, length, format) | Directly measures Su (2026) phantom accuracy phenomenon in our models. |

**Expected outcome**: EXP-006 shows clear position/length shortcuts. EXP-009 shows suppressed shortcuts. This proves our "anti-gaming regularization" thesis.

**Comparison with Su (2026)**: Their paper identifies the problem (85% phantom accuracy in accuracy-trained judges). We provide the solution (multi-objective RL as regularization).

---

### EXP-NEW-2: Post-hoc vs Training-time Head-to-Head

**Goal**: Demonstrate that training-time optimization strictly dominates post-hoc correction, filling a gap identified in the literature (no prior head-to-head comparison exists).

**Protocol**:

| Condition | Description | Inference Cost |
|-----------|-------------|----------------|
| Base model | Qwen2.5-7B-Instruct, single pass | 1x |
| Post-hoc swap (Base) | Base model, eval twice with position swap, majority vote | 2x |
| Post-hoc swap + confidence (Base) | Swap + weighted by confidence, take higher-confidence verdict | 2x |
| EXP-009 Full RL | Our trained model, single pass | 1x |
| EXP-009 + Post-hoc swap | Our trained model with swap averaging on top | 2x |

**Key claims to support**:
1. Full RL (1x cost) matches or beats Post-hoc swap (2x cost) on all metrics
2. Post-hoc swap on top of Full RL yields diminishing returns (the invariance is internalized)
3. The gap between Post-hoc(Base) and Full RL demonstrates that training discovers strategies unavailable to inference-time correction

**Analysis**:
- Compare accuracy: Full RL vs Post-hoc. If Full RL wins, training > patching.
- Compare Full RL vs Full RL + Post-hoc: if gap is small (<1%), invariance is already learned.
- Plot: accuracy vs inference cost curve showing Pareto frontier.

---

### EXP-NEW-3: Per-Category Breakdown

**Goal**: Characterize where consistency/calibration training helps most, connecting to the shortcut suppression mechanism.

**Protocol**:
Evaluate all models (Base, EXP-006, EXP-009) on RewardBench broken down by category:

| Category | Base Acc | Expected Pattern |
|----------|----------|-----------------|
| Chat (general) | ~75% | Moderate gain from consistency (long responses, position bias present) |
| Chat Hard | ~55% | Large gain from consistency (adversarial, shortcuts maximally misleading) |
| Reasoning | ~70% | Small gain from consistency, moderate from calibration |
| Safety | ~90% | Minimal gain (near ceiling, shortcuts less relevant) |
| Code | ~65% | Moderate gain from both (clear correct answers, but formatting exploitable) |

**Specific predictions (testable)**:
1. Categories with highest base consistency already (>90%) should show least improvement from training
2. Categories where base model has high accuracy but low consistency (shortcut-driven accuracy) should show ACCURACY DROP for Acc-only and ACCURACY GAIN for Full
3. Safety category near 1.0 accuracy → minimal improvement (ceiling effect)
4. Chat Hard / adversarial categories → maximum improvement (shortcuts most exploitable here)

**This directly tests the anti-gaming mechanism**: If multi-objective RL works by suppressing shortcuts, its effect should be concentrated in categories where shortcuts are most prevalent (high base accuracy + low consistency = shortcut-driven).

---

### Connection: How New Experiments Support the Paper Narrative

| Paper Claim | Supporting Experiment |
|-------------|---------------------|
| "Accuracy-only RL teaches gaming" | EXP-NEW-1 (shortcut metrics on EXP-006) |
| "Multi-objective RL is anti-gaming" | EXP-NEW-1 (suppressed shortcuts on EXP-009) |
| "Training > post-hoc correction" | EXP-NEW-2 (head-to-head comparison) |
| "Consistency training improves accuracy" | EXP-NEW-3 (per-category: gains where shortcuts existed) |
| "Calibration prevents reward exploitation" | Existing ablation (ECE comparison) + confidence histograms |

### Priority Order for Execution:
1. **EXP-NEW-3** (per-category) — can be done immediately with existing eval results, just needs category-level aggregation
2. **EXP-NEW-1** (reward hacking detection) — requires custom analysis script but uses existing model outputs
3. **EXP-NEW-2** (post-hoc comparison) — requires running base model with swap protocol, moderate cost
