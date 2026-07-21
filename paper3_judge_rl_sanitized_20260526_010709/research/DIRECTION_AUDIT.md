# Direction Audit: Paper 3 — "Shortcut Amplification in Judge RL Training"

**Date**: 2026-05-18
**Auditor**: Direction Audit Agent
**Deadline**: EMNLP ARR May 2026-05-25 AoE (~7 days remaining)

---

## 1. Is the Direction Correct?

### 1.1 "Shortcut Amplification" as General Principle — Assessment

**Verdict: The framing is PARTIALLY correct but overreaches with current evidence.**

The position shortcut finding is strong and well-evidenced (r=1.000, majority vote kill shot). However, claiming a "general principle of RL shortcut amplification" requires more than one carefully validated instance.

**Problems with the "general principle" framing:**

| Issue | Concern |
|-------|---------|
| Only 1 confound fully validated | Position shortcut is the only instance with complete evidence chain |
| Length experiment still running | If it fails or shows different dynamics, generality claim collapses |
| Extrapolation risk | "RL amplifies ANY confound" is a sweeping claim requiring broad evidence |
| Geirhos 2020 already covers supervised | Extending to RL isn't as novel as it sounds — reviewers will ask "obviously?" |

**What 2 instances buys you:**
- Position + length is enough to say "this is not specific to position" and "the mechanism generalizes across confound types"
- It is NOT enough to claim a universal law. Reviewers will point out both confounds are simple (correlational), and ask about more complex confounds
- 2 instances = solid empirical paper; general principle claim needs theoretical backing

### 1.2 Is There a Better Framing?

**Recommended framing: Stay focused on position shortcut + RewardBench as PRIMARY contribution, with length as a generality test.**

The strongest version of this paper is NOT "we prove a general principle" but rather:

> "We reveal that standard RL judge training (RewardBench + accuracy reward) produces phantom accuracy via position shortcutting. This affects ALL published RL-trained judges using RewardBench-format data. We provide diagnostic tools + simple fix."

**Why this framing is stronger:**
1. **Actionable**: Every lab training RL judges on RewardBench is affected
2. **Specific**: Clear scope, not over-claiming
3. **Verifiable**: Anyone can check their own models
4. **Complete story**: problem → diagnosis → mechanism → fix

The "general principle" framing is a discussion section point, NOT the paper's core claim.

### 1.3 Recommended Title Options

| Option | Strengths | Weaknesses |
|--------|-----------|------------|
| "Position Shortcut: How RL Teaches LLM Judges to Cheat" (current) | Memorable, specific, clear | "to Cheat" slightly colloquial for EMNLP |
| "Phantom Accuracy in RL-Trained LLM Judges" | Catchy, hints at Su et al. connection | Less specific about mechanism |
| "The Position Confound in Judge RL Training" | Scientific, precise | Less memorable |
| **"When RL Judges Learn Position Instead of Quality"** | Clear, specific, not over-claiming | Slightly long |

**Recommendation**: Keep current title or use "Phantom Accuracy in RL-Trained Judges: The Position Shortcut" — specific, memorable, falsifiable.

---

## 2. Are Contributions Sufficient for EMNLP Main?

### 2.1 Contribution Audit

| # | Contribution | Type | Strength | EMNLP-worthy? |
|---|-------------|------|----------|---------------|
| C1 | RewardBench position confound identification | Empirical finding | Strong (r=1.000) | ✅ but incremental alone |
| C2 | RL catastrophically amplifies it | Empirical finding | Strong (94%→56% MV) | ✅ key contribution |
| C3 | Diagnostic tools (r, MV, checkpoint dynamics) | Methodology | Medium (tools are simple) | ✅ useful but not deep |
| C4 | Multi-objective rewards can't fix data confound | Negative result | Strong (4 reward modes, 13 runs) | ✅ important |
| C5 | Balanced data fix | Simple solution | Pending validation | ✅ if confirmed |
| C6 | Length confound generality (pending) | Generalization | TBD | ⚠️ nice-to-have |

### 2.2 Honest Assessment

**Is this enough for EMNLP main? YES, but marginally.**

**Strengths:**
- Direct contradiction of JudgeLRM (published, cited 100+ times)
- Complete story: find → diagnose → explain why rewards fail → fix
- Immediately actionable for the community
- Strong evidence quality (multiple seeds, complete ablation)
- Checkpoint dynamics add mechanistic depth

**Weaknesses:**
- Core insight is simple: "biased data → biased model" (the RL amplification adds value, but the fix is just "balance your data")
- Single dataset (RewardBench), single model (Qwen2.5-7B), single algorithm (GRPO)
- Diagnostic tools (r-correlation, majority vote) are ad-hoc, not a principled framework
- "Negative result" papers face reviewer resistance at main conferences

**Comparison to EMNLP 2025 acceptance threshold (22% rate):**
- This is on the boundary between main and Findings
- With balanced results confirming the fix: **main conference viable** (complete story)
- Without balanced results: **Findings-level** (diagnosis without resolution)
- With length confound additionally confirming: **strengthens** the paper but isn't make-or-break

### 2.3 What Would Push This Firmly Into Main

1. **Balanced training results must be strong** — if balanced training restores consistency to ~80%+ while keeping accuracy at ~88%+, this becomes a "discover → fix" paper that's clearly main-worthy
2. **Quantify community impact** — show that X% of recent papers (JudgeLRM, Self-Taught Evaluators, etc.) use RewardBench-format data and are therefore affected
3. **Cross-model validation** — even one additional model (8B or 14B, different family) strengthens enormously

---

## 3. Reviewer Attack Surfaces

### 3.1 Attack: "This is just about data quality, not about RL"

**Severity: HIGH (most dangerous)**

The reviewer argument: "You show biased data → biased model. This is trivially expected. The RL part is irrelevant — any training procedure would exploit perfect position-label correlation. Your contribution reduces to 'clean your data'."

**Defense:**
- Show that the BASE MODEL (without RL) has only mild position preference (80% A-rate on data where gold=A, i.e., it already gets most right by content)
- Show RL AMPLIFIES the shortcut far beyond what the baseline exploits (80%→95% A-rate)
- Show that the amplification is monotonic and accelerates (checkpoint dynamics)
- Argue: "SFT on the same data would not amplify as aggressively because SFT does not have reward maximization pressure"

**Recommended experiment**: Run SFT on the same unbalanced data and show it amplifies LESS than RL. This would be a crucial comparison that distinguishes "data is biased" from "RL + biased data = catastrophic shortcut." **HIGH PRIORITY if time allows.**

### 3.2 Attack: "J1 already knows and fixes this"

**Severity: MEDIUM-HIGH**

J1 (ICLR 2026) uses both-order training and reports position consistency improvements. A reviewer could say "J1 solves this problem already."

**Defense:**
- J1 is a SOLUTION paper that builds a better system. They never DEMONSTRATE the catastrophic failure mode.
- We are a DIAGNOSTIC paper: we show (a) HOW bad it is, (b) WHY rewards can't fix it, (c) the mechanism
- J1 doesn't report the r=1.000 correlation, the majority vote collapse, or the checkpoint dynamics
- Analogy: J1 is the vaccine; we are the epidemiology paper showing the disease
- **Critical**: Explicitly cite J1 and position our work as "the problem that J1's solution addresses, which we characterize comprehensively for the first time"

### 3.3 Attack: "Only two instances doesn't prove general principle"

**Severity: LOW (only applies if you over-claim)**

If you frame the paper as "general RL shortcut amplification law" → this attack is devastating.
If you frame it as "position shortcut in judge RL training, with evidence this mechanism extends to other confounds" → it's just a minor scope limitation.

**Defense**: Don't over-claim. Keep generality in Discussion, not in the title/abstract/contributions.

### 3.4 Attack: "RewardBench is one benchmark, results may not generalize"

**Severity: MEDIUM**

**Defense:**
- RewardBench is THE standard benchmark for reward model/judge evaluation (Lambert et al., 2024)
- The structural confound (chosen always in position A) is NOT unique to RewardBench — it's the STANDARD format for preference datasets (HH-RLHF, UltraFeedback, etc.)
- Check and report: do other popular preference datasets have the same position confound?
- State explicitly: "Any preference dataset that places chosen in a fixed position will produce this shortcut under RL training"

### 3.5 Attack: "The fix is trivial — why is this a paper?"

**Severity: MEDIUM**

**Defense:**
- The fix being simple does NOT make the finding trivial
- JudgeLRM (2025), published with significant impact, did NOT use this fix and reported misleading results
- The community clearly did not know this was a problem (otherwise JudgeLRM wouldn't have been published with those claims)
- The contribution is the DIAGNOSIS, not the fix
- Analogy: Simpson's paradox is a trivial statistical principle, but identifying it in specific real-world datasets has high value

### 3.6 Attack: "Single model (Qwen2.5-7B), single algorithm (GRPO)"

**Severity: MEDIUM-HIGH**

This is a legitimate concern for generalizability.

**Mitigation options (in priority order):**
1. Run one experiment with a different model (e.g., Llama-3-8B or Qwen3-8B) — even 1 seed shows it's not Qwen-specific
2. Argue that GRPO is representative of policy gradient methods (PPO, REINFORCE would show same behavior because the reward structure is identical)
3. Cite Su et al. (2026) who found similar phantom accuracy with different models/methods

---

## 4. Missing Experiments Assessment

### 4.1 MUST-HAVE (without these, paper is incomplete)

| Experiment | Status | Why Critical | Time Needed |
|---|---|---|---|
| Balanced training eval (all 7 runs) | Running (~7-10h) | Without this, no "fix" → paper is only diagnosis | Wait for completion |
| Balanced majority vote analysis | After balanced eval | Must show MV accuracy recovers with balanced training | 30min (zero GPU) |

### 4.2 STRONGLY RECOMMENDED (significantly strengthens paper)

| Experiment | Status | Why Important | Time Needed |
|---|---|---|---|
| SFT baseline on unbalanced data | NOT STARTED | Proves RL amplifies MORE than SFT → addresses biggest reviewer attack | 1 GPU, ~2-3h |
| One different base model (e.g., Llama-3-8B or Qwen3-8B) | NOT STARTED | Addresses "single model" attack | 1 GPU, ~4-5h |
| Survey of preference dataset position distributions | NOT STARTED | Shows the confound is widespread, not RewardBench-specific | Zero GPU, 1h |

### 4.3 NICE-TO-HAVE (but won't block acceptance)

| Experiment | Status | Value |
|---|---|---|
| Length confound experiment | Running (3 seeds) | Supports generality argument in Discussion |
| More benchmarks (MT-Bench, AlpacaEval) | Not started | Nice but RewardBench is the standard |
| More RL algorithms (PPO, DPO) | Not started | Diminishing returns; GRPO is representative |
| Per-category breakdown | Partially available | Adds depth but not required |
| Probing analysis (representation-level) | Not started | Would be very strong but too time-consuming |

### 4.4 Prioritized Experiment Plan (given 7 days)

**Days 1-2 (Today + Tomorrow):**
1. Wait for balanced training to complete (auto-running)
2. Launch SFT baseline on unbalanced data (1 GPU, ~3h)
3. Launch 1 seed of different model (Qwen3-8B or Llama-3-8B) on unbalanced data (1 GPU, ~4h)
4. Survey position distribution in HH-RLHF, UltraFeedback, Chatbot Arena data (zero GPU)

**Day 3:**
1. Eval all balanced runs
2. Run balanced majority vote analysis
3. Eval SFT baseline and different-model run
4. Collect length-confound results

**Days 4-5:**
1. Fill paper tables
2. Create figures (training dynamics, position preference curves)
3. Complete per-category analysis

**Days 6-7:**
1. Polish writing
2. Final compilation + proofread

---

## 5. Timeline Risk Assessment

### 5.1 Critical Path Analysis

```
Day 1-2: Balanced training completes (auto) + SFT baseline + alt model
Day 3: All evals complete → fill tables
Day 4-5: Write remaining sections + figures
Day 6-7: Polish + compile
```

**Bottleneck**: Balanced training completion. If it takes 10h as estimated, results available by tonight/tomorrow morning.

### 5.2 Risk Table

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Balanced training fails (crashes/OOM) | 10% | HIGH | Check logs frequently; restart if needed |
| Balanced training shows unexpected results | 20% | MEDIUM | Pivot narrative; any result is publishable |
| SFT baseline doesn't have time | 15% | MEDIUM | Argue theoretically that RL amplifies more |
| Length experiment shows no amplification | 30% | LOW | Drop from paper; not critical |
| Writing not done in time | 25% | HIGH | Paper draft exists; focus on filling tables |
| GPU contention from other sessions | 20% | MEDIUM | Coordinate; don't need many GPUs for eval |

### 5.3 Honest Timeline Assessment

**Can this paper be SUBMITTED by 2026-05-25? YES, with discipline.**

The paper draft already exists with structure, introduction, and methods complete. The main gaps are:
1. Table 2 (balanced results) — depends on experiments finishing
2. Figures (training dynamics) — data exists, need to plot
3. Per-category table — can be filled from existing data
4. Discussion polish

**Risk level: MODERATE.** The paper is submittable even without length confound and without a second model. The minimum viable paper is:
- Position shortcut finding (DONE)
- Multi-objective failure (DONE)
- Balanced training fix (pending ~24h)
- Training dynamics (data exists, need figure)

---

## 6. Strategic Recommendations

### 6.1 Paper Scope Decision

**RECOMMENDED**: Frame as a focused empirical paper on position shortcut in judge RL training.

**DO NOT**: Over-claim "general RL shortcut amplification principle" — save that for a follow-up or Discussion section.

### 6.2 Contribution Framing (for camera-ready abstract/intro)

1. **Finding**: Standard RL judge training produces "phantom accuracy" — 40% of accuracy gains come from position shortcutting, not genuine judgment (directly contradicts JudgeLRM)
2. **Mechanism**: Training data position confound + RL reward maximization → monotonic shortcut amplification
3. **Failure**: Multi-objective rewards (consistency proxy, calibration) cannot fix data-level confounds
4. **Fix**: Balanced training data removes confound, preserves genuine accuracy gains
5. **Diagnostic toolkit**: Position-swap majority vote as ground-truth capability measure

### 6.3 Key Differentiators vs. Competitors

| Competitor | What they don't have |
|---|---|
| JudgeLRM (2025) | They CLAIM consistency improves; we PROVE it collapses |
| J1 (ICLR 2026) | They build a fix; we characterize the disease with r=1.000, MV collapse, dynamics |
| Silent Judge (NeurIPS-W 2025) | They study prompt shortcuts in inference; we study training-time shortcut creation |
| PRISM (NeurIPS 2025) | They fix reward MODEL shortcuts; we study POLICY training shortcut amplification |
| FairJudge (2026) | Multi-stage pipeline; doesn't isolate position confound as root cause |
| Su et al. (2026) | Documents "phantom accuracy" symptom; we identify specific mechanism + fix |

### 6.4 One-Sentence Positioning

"We are the first to identify position confounding as the specific mechanism behind phantom accuracy in RL-trained judges, demonstrate that reward engineering cannot overcome it, and validate that balanced data resolves it."

---

## 7. Final Verdict

| Dimension | Score | Notes |
|---|---|---|
| Direction correctness | 8/10 | Good direction; just don't over-claim generality |
| Contribution sufficiency | 7/10 (→8/10 with balanced fix) | On boundary; balanced results push to main |
| Evidence quality | 9/10 | r=1.000, 13 models, multiple seeds, checkpoint dynamics |
| Novelty vs. competitors | 7.5/10 | Strong vs JudgeLRM; must clearly differentiate from J1 |
| Timeline feasibility | 7/10 | Tight but doable with discipline |
| Biggest risk | - | "Trivially expected" + "just clean data" reviewer pushback |

### Recommendation: PROCEED with focused position-shortcut framing. Do NOT over-claim general principle.

**Critical actions:**
1. Ensure balanced training completes and eval is done (make or break)
2. Run SFT baseline (addresses biggest reviewer attack at low cost)
3. Keep "general shortcut amplification" in Discussion only, not in title/contributions
4. Explicitly position vs J1 in Related Work (complementary, not competing)
5. Survey position confound prevalence in other datasets (shows community-wide impact)
