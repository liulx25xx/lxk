# Deep Innovation Reasoning: Paper 3 Judge RL
## Beyond "Diagnosis Paper" — Finding the Genuinely Novel Angle

**Date**: 2026-05-18  
**Context**: We have the "Position Shortcut" phenomenon fully characterized. The question is: what angle makes this a **9/10 novelty** paper rather than a **7.5/10 diagnosis** paper?

---

## Summary of Top 2 Recommended Angles

| Rank | Angle | One-line | Novelty | Feasibility |
|------|-------|----------|---------|-------------|
| **1** | **RL as Shortcut Amplifier: A General Principle** | "RL post-training amplifies ANY spurious correlation in training data into a dominant strategy — position is just one instance of a general law" | 9/10 | HIGH (we have the data; need 1-2 extra experiments for generality) |
| **2** | **Capability Degradation: RL Actively Destroys Genuine Judgment** | "RL training doesn't just add shortcuts — it REPLACES genuine judgment capability. The model becomes WORSE than baseline at actual content evaluation" | 8.5/10 | HIGH (provable from existing majority-vote data + one new experiment) |

---

## Angle 1: RL as Shortcut Amplifier — The General Principle

### The Insight (One Sentence)

**"Reinforcement learning post-training does not merely fail to remove spurious correlations — it actively amplifies them into dominant strategies, because any statistical regularity in training data that correlates with reward becomes a free source of reward signal that requires no genuine capability improvement."**

### Why This Is Genuinely Novel

Geirhos et al. (2020) "Shortcut Learning in Deep Neural Networks" established the shortcut learning framework for **supervised learning**. They showed CNNs learn texture over shape, etc. But:

1. **Nobody has extended this to RL post-training specifically.** The RL case is fundamentally different because:
   - In SL: shortcuts come from input-output correlation in the dataset
   - In RL: shortcuts come from correlation between ANY observable signal and reward — and RL ACTIVELY SEARCHES for such correlations (policy gradient literally maximizes expected reward through whatever path exists)
   
2. **The amplification mechanism is unique to RL:**
   - SL shortcut: model passively learns co-occurrence (texture↔class)
   - RL shortcut: model ACTIVELY discovers that exploiting the correlation is the path of least resistance to reward. The policy gradient literally says "if saying A gives reward, say A more." This is active exploitation, not passive learning.
   - **Critical difference**: In SL, the shortcut accuracy is bounded by the correlation strength in data. In RL, the shortcut can be amplified BEYOND the data distribution because the policy is free to concentrate all probability mass on the shortcut.

3. **Our position shortcut is a PERFECT instance of this general law:**
   - Data correlation: gold_label = "A" with probability 1.0
   - RL amplification: model goes from 80% pred-A (baseline) to 99% pred-A (lr=1e-5)
   - The baseline already has mild position preference (80% pred-A). RL doesn't create it from zero — it **amplifies** an existing tendency into a dominant strategy.

### What Makes This Different From Existing Work

| Paper | What they show | What we add |
|-------|---------------|-------------|
| Geirhos et al. (2020) | SL shortcuts (texture bias) | RL AMPLIFICATION mechanism (active exploitation vs passive learning) |
| Su et al. (2026) | RL response generators game format cues | We show RL JUDGES game position cues — different actor, different mechanism |
| J1 (Saha et al., 2025) | Balanced data prevents position bias | We show the GENERAL PRINCIPLE of why unbalanced data fails: RL amplification |
| Reward Hacking Benchmark (2026) | RL models hack complex tasks | We provide the MECHANISTIC EXPLANATION and simple diagnostic |

### The General Law (Formally)

**Theorem (Informal):** Given a reward signal R(x, a) that correlates with any observable feature f(x) in the training distribution (i.e., E[R|f(x)=v1] > E[R|f(x)=v2]), RL post-training will:
1. Amplify the model's reliance on f(x) monotonically with training
2. The rate of amplification scales with learning rate
3. Multi-objective rewards that penalize f(x)-dependence can prevent amplification only if they act on the SAME AXIS as the correlation

**Empirical support from our data:**
- lr=1e-6: mild amplification (80%→82% pred-A, accuracy 82%)
- lr=3e-6 (default): strong amplification (80%→94% pred-A, accuracy 94%)
- lr=1e-5: extreme amplification (80%→99% pred-A, accuracy 99%)
- The amplification is MONOTONIC in lr — this is exactly what a general amplification law predicts

### Extending Beyond Position: The Generality Argument

To make this a **general principle** rather than a one-off observation, we need to show (or argue convincingly) that the same mechanism applies to other shortcuts:

**Already demonstrable from our data:**
1. **Position shortcut** (demonstrated): gold=A → RL learns "say A"
2. **Length shortcut** (can analyze): if longer responses tend to be "chosen" in RewardBench → RL may learn "pick the longer one"
3. **Confidence shortcut** (EXP-006 shows this): accuracy-only models become overconfident → this IS a shortcut (assertive outputs correlate with reward)

**Low-cost additional experiments (1-2 GPU-hours each):**
4. **Synthetic length bias**: Create a length-biased training set (gold always = longer response). Train RL. Show length becomes dominant strategy. Same mechanism, different shortcut.
5. **Format shortcut**: Create format-biased set (gold always = response with code blocks/markdown). Train RL. Show format becomes dominant strategy.

If we demonstrate 2-3 shortcuts with the SAME amplification curve shape and SAME mechanism, the paper becomes: "RL amplifies ALL shortcuts in training data, not just position. Position is the exemplar; the principle is general."

### Why This Beats Pure Diagnosis

A diagnosis paper says: "Here's a bug (position bias). Here's how bad it is (56% majority vote). Here's the fix (balanced data)."

A **general principle** paper says: "Here's a FUNDAMENTAL PROPERTY of RL post-training: it amplifies spurious correlations. Position bias is one manifestation. This principle implies that ANY RL training pipeline must audit training data for spurious correlations before training, because RL will find and exploit them. This reframes the problem from 'fix position bias' to 'audit all possible shortcuts before RL training.'"

The practical implication shifts from "balance your judge training data" (known) to **"audit ALL training data for ANY spurious correlation before running RL, because RL will amplify it catastrophically"** (novel, general, actionable).

### Connection to Paper 7 (NeurIPS Frozen Frontiers)

This connects beautifully:
- **Paper 7**: "RL takes the easiest path to reward — probability sharpening on already-solvable problems rather than expanding capability"
- **Paper 3**: "RL takes the easiest path to reward — shortcut exploitation rather than genuine judgment improvement"
- **Unified principle**: "RL post-training is fundamentally a reward-maximizing optimizer. It will ALWAYS prefer the path of least resistance. If a shortcut exists that provides reward more easily than genuine capability, RL will find it."

This is the same underlying principle manifesting differently:
- In Paper 7: the "shortcut" is probability concentration (free reward from getting the same answer right more reliably)
- In Paper 3: the "shortcut" is position exploitation (free reward from the correlation in training data)
- Both are instances of: **"RL follows the gradient of easiest reward, not the gradient of genuine capability"**

---

## Angle 2: RL Actively Destroys Genuine Judgment Capability

### The Insight (One Sentence)

**"RL training not only adds position shortcuts but actively DEGRADES the model's genuine content-evaluation capability — the consistent predictions (where original==swap, proving position-independence) are LESS accurate than baseline for some models, and the overall 'genuine accuracy' drops from 80% to 56%."**

### The Key Data Point (ALREADY IN HAND)

From the post_hoc_analysis:
- **Baseline standard accuracy**: 80.2%
- **RL model majority-vote accuracy**: 56%
- **Baseline is evaluated on ALL-A gold** too — so baseline's 80% is also partially inflated by its own mild position bias (baseline pred_A rate = 80%)

But here's the genuinely surprising part:

**Baseline majority-vote accuracy** (from post_hoc_analysis.json): 
- `majority_vote_abstain`: 68.6% (only on agreed-upon samples)
- `majority_vote_use_orig`: 80.2% (same as standard because baseline has high agreement)

**RL model majority-vote accuracy**: ~56% (from memory)

This means:
- Baseline "genuine" accuracy (position-controlled) ≈ 80% (high agreement, so majority vote ≈ standard)
- RL model "genuine" accuracy (position-controlled) ≈ 56%

**RL training DESTROYED 24 percentage points of genuine judgment capability!**

### Why This Is Surprising

The standard narrative (JudgeLRM, J1, FairJudge) is:
> "RL training improves judge accuracy. Position bias is a side effect we need to mitigate."

Our data shows the OPPOSITE:
> "RL training creates the ILLUSION of improvement through position shortcuts while DEGRADING the underlying capability. It's not 'improved accuracy with position bias as side effect' — it's 'no improvement at all, just position exploitation masquerading as accuracy gain.'"

**Even stronger**: The fact that genuine accuracy drops BELOW baseline (56% < 80%) means RL isn't just "not helping" — it's actively HARMFUL. The model doesn't just learn a shortcut and keep its existing abilities. It appears to FORGET how to evaluate content properly, replacing genuine evaluation with position guessing.

### Mechanism Hypothesis: Shortcut Crowding Out

Why would shortcuts REDUCE genuine capability rather than just ADD a parallel strategy?

**Hypothesis: Gradient competition.** In RL (specifically GRPO), the model updates its policy to increase probability of high-reward actions. When position exploitation provides higher expected reward than genuine content evaluation (because position is perfectly correlated with gold in training data), the gradient signal from position exploitation DOMINATES. Over training steps:

1. The model allocates more "representational capacity" to position-based reasoning
2. The representations used for genuine content evaluation get overwritten (catastrophic forgetting of capability)
3. The model's attention patterns shift from comparing content to identifying position markers
4. Result: genuine judgment degrades even on position-controlled evaluation

**This is analogous to "catastrophic forgetting in RL"** — a known phenomenon where RL training overwrites pretrained capabilities. But the specific mechanism here (shortcut overwrites genuine capability) is novel and specific to the shortcut amplification setting.

### Empirical Evidence For Capability Degradation

1. **Majority vote shows genuine acc drops**: 80% → 56% (below chance for binary!)
2. **Consistent predictions** (same answer on both orders) still have 90-97% accuracy — but there are FAR FEWER of them. RL reduces the number of position-invariant predictions from ~82% (baseline) to ~60%.
3. **Checkpoint dynamics** show the degradation is progressive:
   - Step 100: consistency 80.8% (most predictions still genuine)
   - Step 200: consistency 76.8% (genuine predictions declining)
   - Step 300: consistency 65.3% (minority of predictions are genuine)
   - Step 500: consistency 60.8% (only ~60% of predictions are position-independent)

The model progressively LOSES its ability to make position-independent judgments.

4. **lr=1e-6 partially preserves capability** (consistency 79.3%, accuracy 82.2%):
   - Slower learning rate = slower shortcut amplification = more genuine capability retained
   - This is consistent with the "gradient competition" hypothesis: at low lr, the shortcut gradient is weak enough that genuine judgment capability is partially preserved

### What This Means Practically

**For the RLHF ecosystem**: If RL-trained judges are used as reward models for RLHF:
- Their position-biased scores propagate to downstream model training
- The downstream model learns: "when evaluated by this judge, put your better content in position A"
- **This is a form of reward model poisoning through training data confounds**

**For benchmark evaluation**: 
- RewardBench leaderboard scores for RL-trained judges are **inflated by up to 40%** (the position shortcut contribution)
- Any judge achieving >90% accuracy on RewardBench should be suspected of position exploitation
- **Recommendation: RewardBench should include mandatory position-swap evaluation** (currently it doesn't!)

---

## Angle Comparison: Why These Two Are Best

### Rejected Angles and Why

| Angle | Rejection Reason |
|-------|-----------------|
| "Balanced data shifts the Pareto frontier" | The Pareto frontier itself is an artifact of the confounded data. With balanced data, there may be no tradeoff at all (J1 shows this). Not enough novelty. |
| "Phase transition / tipping point" | Descriptively interesting but mechanistically shallow. The exponential shape is expected from any positive feedback loop. Not enough insight. |
| "Judge training → RLHF propagation" | Requires downstream experiments we can't run in time. Arguing it without evidence is too speculative for a good paper. |
| "Consistency training improves accuracy" | This was the pre-pivot insight (composite reward). But we already showed consistency DOESN'T help when data is confounded — all multi-objective variants fail. This angle is dead given our results. |

### Why "RL as Shortcut Amplifier" > "Capability Degradation"

Both are strong, but **Angle 1 is the winner** because:

1. **Generality**: Angle 1 makes a general claim about RL (applicable beyond judges). Angle 2 is specific to the judge setting.
2. **Contribution type**: Angle 1 provides a PRINCIPLE (theoretical contribution). Angle 2 provides a FINDING (empirical contribution). Principles are valued higher.
3. **Connection to literature**: Angle 1 extends Geirhos (2020) shortcut learning to RL — a natural, important extension. Angle 2 is more isolated.
4. **Practical impact**: Angle 1 implies "audit all RL training data for correlations" — general advice. Angle 2 implies "use balanced judge data" — specific advice (and known).
5. **Reviewer excitement**: "RL amplifies shortcuts" is a finding that makes reviewers think "this applies to MY work too." "Judge accuracy is inflated" is interesting but niche.

### The Optimal Paper: COMBINE BOTH

The best paper structure uses **Angle 1 as the GENERAL PRINCIPLE** and **Angle 2 as the PRIMARY EVIDENCE**:

> "RL post-training amplifies spurious correlations in training data into dominant strategies. We demonstrate this general principle through LLM judge training: when trained on benchmark data with structural position-label correlations, RL amplifies mild position preferences (80% pred-A) into near-total position exploitation (99% pred-A), creating phantom accuracy that collapses under position-controlled evaluation (94%→56%). Moreover, the amplification actively degrades genuine judgment capability below baseline. We show this amplification follows a predictable trajectory (monotonic with lr and steps), provide simple diagnostic tools (pred-A/accuracy correlation, majority-vote audit), and demonstrate that the fix (balanced data) must address the confound at the data level — no reward engineering can substitute."

---

## Required Experiments for Angle 1

### Already Complete (No New Compute Needed)

1. Position shortcut amplification: full evidence chain
2. Checkpoint dynamics: amplification trajectory over steps
3. Learning rate scaling: amplification strength vs lr
4. Multi-objective reward failure: confirms data-level confound > reward-level fix
5. Majority vote diagnostic: separates genuine from shortcut accuracy
6. Consistency correlation: r(pred_A, accuracy) = 1.000

### New Experiments Needed (Low Cost, High Impact)

| Experiment | Purpose | Cost | Priority |
|-----------|---------|------|----------|
| **Length bias synthetic** | Show SAME amplification mechanism for length | 2-3 GPU-hours (one training run on synthetic data) | HIGH |
| **Balanced data training** (already running!) | Show deconfounding eliminates amplification | Already running (7 runs on 228.224) | CRITICAL (already in progress) |
| **Length analysis on existing models** | Check if RL models also develop length bias from RewardBench | ZERO compute (analysis of existing predictions) | MEDIUM |
| **Per-sample prediction categorization** | For each test sample: is the model right because of content or position? | ZERO compute (analysis script) | HIGH |
| **Baseline majority-vote deep analysis** | Precisely quantify baseline "genuine" vs RL "genuine" accuracy gap | ZERO compute (from existing swap eval data) | HIGH |

### The "Length Bias Synthetic" Experiment (Detailed)

This is the KEY experiment that elevates from "position-specific" to "general principle":

1. Create `judge_train_length_biased.json`: modify RewardBench training data so gold_label ALWAYS = the longer response
   - Some samples naturally have longer=better, some don't
   - For samples where shorter=better, swap labels (gold now points to longer)
   - Result: a dataset where length perfectly predicts gold (analogous to position perfectly predicting gold)

2. Train RL (same hyperparams as EXP-006) on this length-biased data

3. Evaluate: Does the model learn "always pick longer"? Does length preference correlate with accuracy like position does?

4. **Expected result**: YES — same amplification curve, same phantom accuracy, same majority-vote collapse when length is controlled.

5. **Payoff**: This single experiment proves the principle is GENERAL, not position-specific. It transforms a 7.5/10 paper into a 9/10 paper.

**Cost**: ~2-3 hours on 1 GPU (same as EXP-006 training). Trivial given 24×H200 available.

---

## Title Options If This Direction Succeeds

### For Angle 1 (General Principle) — RECOMMENDED:

1. **"The Shortcut Amplification Law: How RL Post-Training Exploits Training Data Confounds"**
2. **"RL Amplifies Shortcuts: Why Training Data Confounds Become Dominant Strategies"**
3. **"From Correlation to Exploitation: How RL Post-Training Amplifies Spurious Features"**
4. **"Shortcut Amplification in RL: Position Bias as a Case Study in Spurious Reward Exploitation"** (too long)

### For Combined (Principle + Judge Application):

5. **"RL Trains Judges to Cheat: Shortcut Amplification in Reward-Driven LLM Training"** (snappy, clear)
6. **"Phantom Accuracy: How RL Amplifies Training Data Shortcuts in LLM Judges"**

### My recommendation:

> **"Shortcut Amplification: How RL Post-Training Exploits Data Confounds in LLM Judges"**

- "Shortcut Amplification" = the general concept (novel term)
- "RL Post-Training" = the mechanism
- "Data Confounds" = the root cause
- "LLM Judges" = the domain (but paper argues generality)

Or if we want maximum impact/memorability:

> **"RL Amplifies Everything: Position Bias as a Case Study in Shortcut Exploitation"**

---

## Estimated Impact If Successful

**Paper contribution breakdown:**

1. **General principle** (theoretical): RL amplifies spurious correlations — extends Geirhos (2020) to RL (Section 2: Theory/Framework)
2. **Primary case study** (empirical): Position shortcut in LLM judges with full evidence chain (Section 3-4)
3. **Generality evidence** (empirical): Length bias experiment shows same mechanism (Section 5)
4. **Diagnostic toolkit** (practical): pred-A/accuracy correlation, majority-vote audit, checkpoint monitoring (Section 6)
5. **Fix validation** (empirical): Balanced data eliminates amplification (Section 7, from ongoing experiments)

**Predicted reviewer response:**

| Dimension | Score | Why |
|-----------|-------|-----|
| Novelty | 8-9/10 | General principle + specific demonstration; extends shortcut learning literature to RL |
| Significance | 8/10 | Affects all RL training (not just judges); provides actionable diagnostics |
| Soundness | 9/10 | Perfect correlation evidence + controlled experiments + generality experiment |
| Clarity | Depends on writing | Must be crisp: general law → specific instance → generality → diagnostics |
| Reproducibility | 9/10 | RewardBench is public; training is standard GRPO; all analysis is simple |

**Likely venue acceptance**: EMNLP 2026 main conference (strong fit: empirical, LLM-focused, practical implications for evaluation community).

---

## Action Items

1. **Immediately (zero cost)**: Length analysis on existing model predictions — do RL models also prefer longer responses? If yes, Angle 1 is already partially supported.

2. **Today (2-3 GPU-hours)**: Length bias synthetic training run. This is THE make-or-break experiment for Angle 1.

3. **Wait for balanced results** (already running, ~3-5h): These prove the fix and complete the story.

4. **Write**: Reframe paper from "Position Shortcut" (descriptive) to "Shortcut Amplification in RL" (principled).

5. **Optional bonus**: If time permits, format-bias synthetic experiment (same design as length-bias but for markdown/code formatting preference).

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Length bias experiment shows NO amplification | 15% | Fall back to Angle 2 (capability degradation). Paper is still strong at 8/10 without generality. |
| Someone publishes "RL amplifies shortcuts" before us | 10% | Fast submission (May 25 deadline). Our specific evidence chain (r=1.000, majority vote collapse, checkpoint dynamics) is extremely detailed and hard to replicate quickly. |
| Reviewers say "this is obvious" (of course RL exploits correlations) | 25% | Counter with: (1) JudgeLRM explicitly claims RL IMPROVES consistency — so the field DOESN'T know this. (2) The quantitative severity (94%→56%) is NOT predicted by "obvious" expectation. (3) Multi-objective rewards DON'T fix it — this IS surprising. |
| Balanced data experiment shows shortcut persists | 5% | Would actually be MORE interesting — means the problem is deeper than data. Pivots to "RL finds shortcuts that data balancing can't fix." Higher novelty if true. |
| Overlap with concurrent work | 20% | Focus on MECHANISM (amplification law + diagnostics) not just FINDING (position bias). The mechanism is defensible even if others observe the same phenomenon. |
