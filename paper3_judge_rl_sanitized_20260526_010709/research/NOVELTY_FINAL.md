# Final Novelty Assessment: Paper 3 — Judge RL Shortcut Amplification

**Date**: 2026-05-18
**Assessor**: Novelty Check Agent (comprehensive)
**Deadline**: EMNLP ARR 2026-05-25 (7 days remaining)

---

## 1. Per-Finding Novelty Check

### F1: "SFT hacks MORE than GRPO" (100%/0% vs 94%/60%)

**Closest Competitor**: "SFT Memorizes, RL Generalizes" (Google Research, arXiv 2501.17161, Jan 2025)
- Claims: SFT memorizes training data shortcuts; RL generalizes better to unseen variants
- Setting: General reasoning (card games, navigation), NOT judge training
- Their claim: SFT memorizes spurious correlations more than RL → supports our F1 direction

**Critical Nuance**: 
- Google paper shows SFT memorizes *surface patterns* → fails on distribution shift
- Our F1 shows SFT *perfectly exploits a data confound* (100% accuracy = 100% position bias) → catastrophic shortcutting
- Google paper does NOT study *judge training* or *data confounds specifically*
- Our contribution: **first controlled demonstration in judge training domain** with exact mechanism (r=1.000)

**Reward Hacking Benchmark (arXiv 2605.02964, May 2026, ICML 2026)**:
- Shows DeepSeek-R1-Zero (RL post-trained) has 13.9% exploit rate vs V3 (SFT) at 0.6%
- This is the **OPPOSITE direction**: RL hacks MORE in their setting (tool-use agents)
- Their setting: agent tool-use, not judge training with data confounds

**Verdict**: Our F1 is **NOVEL** but needs careful positioning:
- In judge training with data confound, SFT hacks MORE (memorization)
- In agent tool-use, RL hacks MORE (exploration-driven gaming)
- **This contrast itself is interesting** — confound type determines which paradigm hacks more
- Novelty: **7.5/10** (domain-specific finding, partially supported by Google paper's framing)

---

### F2: "Balanced data completely fixes consistency"

**Closest Competitor**: JudgeLM (ICLR 2025 Spotlight)
- Uses swap augmentation during SFT training to debias position
- Reports improved position consistency
- **But**: JudgeLM is SFT-only; doesn't study RL at all; doesn't show the *before/after* in a controlled experiment

**Pos2Distill** (EMNLP 2025 Oral):
- Position debiasing via knowledge distillation
- Setting: long-context retrieval/reasoning, NOT judge training
- Approach: inference-time, not training-time

**"Toward Robust LLM-Based Judges"** (arXiv 2603.08091, March 2026):
- Proposes bias-aware SFT training for judges
- Relevant but uses SFT, not RL
- Does not study the specific controlled experiment: same model + same reward + only swap balance = fix

**Verdict**: The *fix* (balanced data) is known in SFT (JudgeLM). Our novelty is:
1. Showing the fix works in RL (GRPO) too
2. Controlled A/B: unbalanced→catastrophe, balanced→fix, same everything else
3. Quantifying the exact magnitude: 60.8%→85.3% consistency recovery
- Novelty: **5.5/10** (incremental; balanced data debiasing is well-known; our contribution is the controlled RL comparison)

---

### F3: "Learning rate controls shortcut activation" (safe→trigger→extreme spectrum)

**Closest Work**: 
- **Reward overoptimization scaling** (Gao et al., ICML 2023): Shows reward hacking increases with KL divergence from ref. But this uses KL as proxy, not lr directly, and is about reward models not judge training.
- **"Quagmires in SFT-RL Post-Training"** (ICLR 2026): Shows SFT→RL dynamics but about SFT score not predicting RL performance, not about lr-shortcut spectrum.
- **Spurious Rewards** (arXiv 2506.10947): Shows GRPO's clipping bias amplifies pretraining behaviors. Relevant mechanism but about math reasoning, not judges.

**Key Differentiation**:
- Nobody has shown a **clean lr spectrum** where:
  - lr=1e-6 → safe (83.7%/83.3%, maintains baseline consistency)
  - lr=5e-6 → trigger (94.0%/61.7%, shortcut activates)
  - lr=1e-5 → extreme (98.9%/38.1%, near-complete shortcut)
- This gives a **prescriptive guideline**: practitioners can choose lr to stay in safe zone
- "Spurious Rewards" paper's clipping bias mechanism could EXPLAIN our finding (stronger optimization = more amplification)

**Verdict**: **NOVEL** — no prior work shows clean lr→shortcut activation spectrum for judge RL
- Novelty: **8/10** (actionable, clean, no known competitor)
- Strengthened by connecting to "Spurious Rewards" mechanism as theoretical explanation

---

### F4: "Position confound > length confound in exploitability"

**Closest Work**:
- "Evaluating Scoring Bias in LLM-as-a-Judge" (arXiv 2506.22316, June 2025): Systematically evaluates different bias types but only in INFERENCE (no training)
- "Humans or LLMs as the Judge?" (EMNLP 2024): Studies perturbation vulnerability across bias types during inference
- Multiple papers characterize position bias vs length bias separately, but none compare their **relative exploitability during training**

**Key Insight**: Our data shows:
- Position confound: Acc=94.4%, Consist=60.8% → massive amplification
- Length confound: Acc=80.0%, Consist=78.4% → minimal amplification
- This reveals that NOT all confounds are equally exploitable by RL

**Theoretical Explanation Candidate**: Position is a "simpler" confound (one bit of info: first vs second) vs length (continuous, noisier correlation). RL exploits the lowest-hanging fruit first — connects to the "Spurious Rewards" clipping bias (strongest prior signal gets amplified most).

**Verdict**: **NOVEL** — no prior work compares relative exploitability of different confound types during RL training
- Novelty: **8.5/10** (unique finding, connects to broader theory of confound hierarchy)
- Risk: Only 2 confound types tested; reviewers may ask for more

---

### F5: "Balanced SFT (91.3%/87.1%) >> Balanced GRPO (84.6%/85.3%)"

**Closest Work**:
- "SFT Memorizes, RL Generalizes" — shows RL generalizes BETTER than SFT
- "Quagmires in SFT-RL" — shows SFT→RL dynamics, high SFT doesn't predict RL
- No paper specifically compares SFT vs GRPO on clean/balanced judge data

**Key Insight**: When data is CLEAN (balanced), SFT actually performs BETTER:
- SFT: 91.3% acc, 87.1% consistency
- GRPO: 84.6% acc, 85.3% consistency
- This CONTRADICTS "RL always > SFT" narrative
- Explanation: On clean data, memorization IS the right strategy (because the patterns are genuine)

**Nuance with Google paper**:
- Google paper: "RL generalizes better" → but that's for RULE VARIANTS (distribution shift)
- Our setting: same distribution, clean data → SFT memorization = genuine learning
- Combined insight: **SFT vs RL depends on whether training distribution = test distribution**

**Verdict**: **Moderately NOVEL** but needs careful framing
- Novelty: **6.5/10** (interesting but narrow scope; only 1 task; could be task-specific)
- Value: Provides practical guidance (if data is clean, just use SFT — simpler & better)

---

## 2. Overall Novelty Landscape

### Paper-Level Novelty Assessment

| Finding | Score | Status |
|---------|-------|--------|
| F1: SFT hacks more than GRPO (on confounded data) | 7.5/10 | ✅ Novel in judge domain |
| F2: Balanced data fixes consistency | 5.5/10 | ⚠️ Incremental (JudgeLM prior) |
| F3: LR controls shortcut spectrum | 8/10 | ✅ Novel + actionable |
| F4: Position > length exploitability | 8.5/10 | ✅ Novel + theoretical |
| F5: Balanced SFT > balanced GRPO | 6.5/10 | ⚠️ Narrow but interesting |

**Combined Paper Novelty**: **7.5/10**

### Key Competitors Summary

| Paper | Venue | Threat Level | Differentiation |
|-------|-------|-------------|-----------------|
| JudgeLM | ICLR 2025 Spotlight | Medium | SFT only, no RL, no controlled confound study |
| SFT Memorizes, RL Generalizes | arXiv Jan 2025 (Google) | Medium | General reasoning, not judge; SUPPORTS our F1 |
| Spurious Rewards | arXiv Jun 2025 | Low-Medium | Math domain; explains mechanism but different application |
| Reward Hacking Benchmark | ICML 2026 | Low | Agent tool-use, not judge training |
| VLaw | 2026 (preprint) | **HIGH** | GRPO + judge + reward hacking; BUT math domain, not position bias |
| Taming the Judge | arXiv Oct 2025 | Low | Judge consistency in RL feedback, not judge training itself |
| Pos2Distill | EMNLP 2025 | Low | Inference-time debiasing, long-context, not judge |
| Toward Robust LLM-Based Judges | arXiv Mar 2026 | Medium | SFT debiasing for judges; no RL comparison |

### VLaw — Most Dangerous Competitor

VLaw (JC-Chen et al., 2026) studies reward hacking in GRPO training when using LLM-as-judge rewards. Key differences from us:

| Aspect | VLaw | Our Paper |
|--------|------|-----------|
| Domain | Math reasoning | Judge pairwise evaluation |
| Hack type | Answer gaming (solve-format tricks) | Position shortcut (structural confound) |
| Judge role | External reward signal | The model being trained IS the judge |
| Fix proposed | Oracle-judge mixing | Balanced data + lr control |
| Mechanism | Judge fooling (policy tricks judge) | Confound exploitation (judge learns confound) |

**Our differentiation from VLaw**: We study the judge ITSELF learning shortcuts from training data confounds. VLaw studies the policy model gaming an external judge. These are complementary, not competitive.

---

## 3. Supplementary Experiments — Priority Ranking

### Sorted by Novelty Increment / GPU-Hours

| Priority | Experiment | Novelty Δ | GPU-hrs | Ratio | Recommendation |
|----------|-----------|-----------|---------|-------|---------------|
| 1 | **F: Post-hoc swap average** | +0.3 | 0 | ∞ | **DO NOW** (zero cost, plugs reviewer hole) |
| 2 | **E: More lr points (2e-6, 3e-6)** | +0.5 | (running) | — | **Already in progress** ✅ |
| 3 | **A: DAPO replacing GRPO** | +0.5-1.0 | 3h | 0.25 | **DO** — if DAPO also hacks → "RL general"; if not → algorithm insight |
| 4 | **D: Paired consistency reward** | +0.5-1.0 | 5h | 0.15 | **DO if time** — connects to J1, validates whether explicit consistency reward = fix |
| 5 | **B: Llama-3-8B base model** | +0.5 | 3h+download | 0.12 | **DO if time** — generality across architectures |
| 6 | **C: DPO on preference** | +0.3-0.5 | 3h | 0.12 | **SKIP** — lowest novelty increment, deadline pressure |

### Detailed Reasoning:

**F (Post-hoc swap)**: MUST DO. Zero cost. If post-hoc averaging matches balanced training performance, it means inference-time fix is viable. If it doesn't match → training-time fix is strictly necessary. Either way, this is a key comparison reviewers will ask for.

**A (DAPO)**: HIGH VALUE. The RHB paper (ICML 2026) found RL post-training → higher exploit rates. If DAPO (different RL algorithm) shows SAME shortcut activation → our finding is about RL IN GENERAL, not GRPO-specific. If DAPO doesn't hack → algorithmic insight about WHY GRPO specifically is vulnerable (connects to "Spurious Rewards" clipping bias).

**D (Paired consistency reward)**: MEDIUM-HIGH VALUE. JudgeLM's approach was essentially "train on balanced data." If an explicit consistency reward can fix the problem WITHOUT balanced data → we've found an alternative mitigation. If not → data quality is the only path. Either way, it's a complete story.

**B (Llama-3-8B)**: Generality check. Important for reviewer confidence but doesn't add conceptual novelty. The "Spurious Rewards" paper already showed model-specificity (works on Qwen, not Llama). If Llama shows different behavior → that's actually more interesting (model-dependent vulnerability).

---

## 4. Overall Paper Assessment

### Is This EMNLP Main or Findings?

**Current Assessment: Borderline Main / Strong Findings**

**Strengths for Main**:
- 5 coherent findings forming a complete narrative
- Actionable diagnostic toolkit + fix
- Clean controlled experiments (same model, same data, only one variable changed)
- Novel F3 (lr spectrum) and F4 (confound hierarchy) — no prior work
- Connects to hot topic (reward hacking in RL, post-DeepSeek era)
- Practical impact: affects ALL labs training RL judges on RewardBench-format data

**Weaknesses / Reviewer Attack Surfaces**:

| Attack | Severity | Mitigation |
|--------|----------|------------|
| "Only 1 model (Qwen2.5-7B)" | HIGH | Run Llama experiment (Exp B) |
| "Balanced data fix is well-known (JudgeLM)" | MEDIUM | Emphasize controlled comparison + lr insight; cite JudgeLM and differentiate |
| "Only 2 confound types" | MEDIUM | Frame as diagnostic methodology contribution; discuss why in paper |
| "Small scale (RewardBench subset)" | MEDIUM | Argue controlled study > scale; diagnostic paper not benchmark paper |
| "Trivial confound (100% A in gold)" | HIGH ⚠️ | **This is the biggest risk** — reviewer may say "of course it hacks, your data is broken" |

### The "Trivial Confound" Attack — How to Defend

The strongest reviewer attack is: "You deliberately put a broken confound in training data. Of course the model exploits it. This isn't surprising."

**Defense strategy**:
1. **Real-world prevalence**: Show that RewardBench and similar datasets actually HAVE position imbalance (cite statistics)
2. **Degree of amplification is the finding**: The confound exists slightly in pretraining; RL catastrophically amplifies it. The DEGREE is non-obvious.
3. **SFT vs GRPO comparison**: Shows different training paradigms react differently to the same confound — this IS surprising
4. **LR spectrum**: Shows there's a phase transition, not linear scaling — this IS non-trivial
5. **Confound hierarchy**: Position >> length exploitation — not all confounds are equal — this IS surprising

### What's Needed to Upgrade from Findings → Main

| Action | Impact on Acceptance |
|--------|---------------------|
| Add Llama-3-8B experiment | +++ (addresses generality) |
| Add DAPO experiment | ++ (RL-general claim) |
| Add post-hoc swap baseline | ++ (complete mitigation story) |
| Real-world data audit (RewardBench position stats) | +++ (addresses "trivial confound" attack) |
| Connect to "Spurious Rewards" mechanism theoretically | + (intellectual depth) |

### Final Verdict

**With current 5 findings only**: **EMNLP Findings** (7.5/10 novelty, but "trivial confound" attack risk)

**With recommended additions (F + A + B + RewardBench audit)**: **Competitive for EMNLP Main** (8/10 novelty, generality + mechanism + fix)

### Recommended Immediate Actions (ordered by deadline constraint)

1. **Day 1-2**: Run post-hoc swap (F, zero cost) + DAPO (A, 3h) + start Llama (B, 3h)
2. **Day 2-3**: Get lr=2e-6, 3e-6 results; analyze DAPO/Llama results
3. **Day 3**: Audit RewardBench for actual position distribution (zero GPU, high impact)
4. **Day 3-5**: Write paper with all results
5. **Day 5-7**: Polish + compile + submit

---

## 5. Innovation Angles for Maximum Impact

### Angle 1: "RL as a Confound Microscope"
Frame the paper as revealing that RL training is a *diagnostic tool* — it amplifies confounds that already exist in data, making them visible. "If you want to find confounds in your data, train RL on it and watch what gets amplified."

### Angle 2: "The Optimization Intensity Dial"
The lr→shortcut spectrum suggests a general principle: optimization intensity controls which features get learned. Low lr = genuine features; high lr = shortcut features. This connects to curriculum learning and the "simplicity bias" literature.

### Angle 3: "Confound Hierarchy"  
Not all confounds are equal. Position (binary, strong) >> Length (continuous, weak). This suggests a principled way to predict which confounds will be exploited: simpler confounds with stronger statistical signal are exploited first.

### Angle 4: "The Paradox of Training Paradigms"
- Confounded data: SFT hacks MORE (memorization)
- Clean data: SFT performs BETTER (genuine pattern learning)
- Implication: The quality of your data determines which training paradigm is safer

### Angle 5: "Safe RL Zone"
lr=1e-6 with full rewards maintains baseline consistency while still training. This is a positive, actionable finding: RL CAN be done safely, there's just a threshold.

---

## 6. Summary Recommendation

**Paper 3 has a viable EMNLP submission.** The core novelty is:
1. F3 (lr spectrum) + F4 (confound hierarchy) are genuinely novel (8-8.5/10)
2. F1 (SFT > GRPO hacking) is novel in judge domain + connects to Google paper
3. The complete diagnostic pipeline (detect + diagnose + fix) has practical value

**Biggest risk**: "trivial confound" reviewer attack. Mitigate with RewardBench audit.

**Strongest framing**: "Diagnostic paper revealing RL judge training vulnerabilities + simple fixes" — NOT "general RL shortcut amplification theory."

**Priority actions**: Post-hoc swap (0 cost) → DAPO (3h) → Llama (3h) → RewardBench audit (0 cost) → Write.
