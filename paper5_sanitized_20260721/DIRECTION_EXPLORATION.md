# Paper 2 Direction Exploration: Deep Novelty Check & Alternatives

**Date**: 2026-05-16  
**Goal**: Verify novelty of "pseudo-verifier reliability" direction, explore alternatives, recommend best path.

---

## Part 1: Novelty Verdict for "Pseudo-Verifier Reliability Study"

### 1.1 Critical Overlapping Papers Found

The REPOSITION doc claimed "NOBODY" has studied pseudo-reward noise × domain × RLVR training dynamics. **This is no longer accurate.** Several papers from Jan–May 2026 directly attack this space:

#### THREAT LEVEL: HIGH (Direct Overlap with Contribution 2)

| Paper | Date | Overlap |
|-------|------|---------|
| **"Rate or Fate? RLVεR"** ([arXiv:2601.04411](https://arxiv.org/abs/2601.04411)) | Jan 2026 | Develops a **complete theoretical framework** for noisy rewards in RLVR. Proves a sharp phase transition via Youden's index J=TPR-FPR: J>0 → learning (noise only slows convergence); J≤0 → collapse. Validated on code tasks. **Directly covers our Contribution 2's "critical noise threshold" story.** |
| **"Delay, Plateau, or Collapse"** ([arXiv:2605.02909](https://arxiv.org/abs/2605.02909)) | Apr 2026 | Studies **systematic verification errors** on RLVR. Shows false positives cause sub-optimal plateaus to full collapse. Concludes outcomes depend on **error patterns, not just error rates**. Experiments on arithmetic tasks. **Directly covers our Contribution 2's noise injection experiments.** |

#### THREAT LEVEL: MEDIUM (Partial Overlap with Contributions 1 & 3)

| Paper | Date | Overlap |
|-------|------|---------|
| **RESTRAIN** ([arXiv:2510.02172](https://arxiv.org/abs/2510.02172)) | Oct 2025 | Addresses **spurious majorities** in pseudo-label voting. Uses pseudo-label weighting + negative rollout penalization. Nearly matches gold-label performance. **Directly overlaps with Contribution 3's confidence-weighted / rejection-based rewards.** |
| **JURY-RL** ([arXiv:2604.25419](https://arxiv.org/abs/2604.25419)) | Apr 2026 | Addresses **false positives in majority voting** for label-free RLVR. Uses formal verification (Lean) as backup + "ResZero" fallback for unverified consensus. **Overlaps with Contribution 3's calibration mechanism.** |
| **"When Does Verification Pay Off?"** ([arXiv:2512.02304](https://arxiv.org/abs/2512.02304)) | Dec 2025 | **37 models, 9 benchmarks** studying when verification helps. Finds cross-family verification > self-verification. Some domains more amenable. **Partially overlaps with Contribution 1's cross-domain verification audit.** |
| **"Crossing the Reward Bridge"** ([arXiv:2503.23829](https://arxiv.org/abs/2503.23829)) | Mar 2025 | Extends RLVR to medicine, psychology, economics. Uses 7B LLM-based generative scorer. Shows binary verification has high consistency when expert references exist. **Overlaps with Contribution 1's multi-domain scope.** |

#### THREAT LEVEL: LOW (Tangential but Relevant)

| Paper | Date | Relevance |
|-------|------|-----------|
| **Med-RLVR** ([arXiv:2502.19655](https://arxiv.org/abs/2502.19655)) | Feb 2025 | RLVR in medical domain, shows it works for medical QA. |
| **Open-Medical-R1** ([arXiv:2504.13950](https://arxiv.org/abs/2504.13950)) | Apr 2025 | Data selection strategies for medical RLVR. |
| **Self-Harmony** ([arXiv:2511.01191](https://arxiv.org/abs/2511.01191)) | Nov 2025 | Paraphrasing + harmonic mean to stabilize pseudo-labels. |
| **RoiRL** ([arXiv:2510.02892](https://arxiv.org/abs/2510.02892)) | Oct 2025 | Offline alternative to TTRL, trains 2.5× faster. |

### 1.2 Per-Contribution Novelty Assessment

| Contribution | Proposed | Existing Coverage | Remaining Novelty |
|-------------|----------|-------------------|-------------------|
| **C1**: Pseudo-verifier reliability audit across domains | Measure SC accuracy vs ground truth across Math/Science/Law/Medicine/Code/Commonsense | "When Does Verification Pay Off?" covers 9 benchmarks. "Crossing the Reward Bridge" covers diverse domains. Med-RLVR/Open-Medical-R1 cover medicine. | **Moderate**: Nobody specifically measures majority-voting-as-pseudo-reward accuracy per difficulty level within NLP domains (law, medicine, commonsense). But the framing feels incremental given existing work. |
| **C2**: Quantify reward noise impact → critical threshold | Inject 10-50% noise, find collapse threshold | **RLVεR provides the theory** (Youden's index). **"Delay, Plateau, Collapse" does the experiments** (arithmetic). | **LOW**: The core finding (noise causes collapse above threshold) is established. Our version would be "same finding on more domains" — clearly incremental. |
| **C3**: Calibrated Pseudo-Rewards (CPR) fix | Confidence-weighted/rejection-based training | **RESTRAIN** does pseudo-label weighting + self-penalization. **JURY-RL** uses formal verifier + fallback. | **LOW-MODERATE**: The confidence-weighting idea is no longer novel. Our specific formula (agreement × entropy × cross-model) hasn't been published, but the concept is covered. |

### 1.3 Overall Verdict

> **The pseudo-verifier reliability direction is SIGNIFICANTLY more occupied than assumed in REPOSITION.md.**
>
> - Contribution 2 (noise impact study) is essentially **scooped** by RLVεR + "Delay, Plateau, Collapse"
> - Contribution 3 (calibrated rewards) is **largely scooped** by RESTRAIN + JURY-RL
> - Contribution 1 (reliability audit) retains **moderate novelty** but feels incremental alone
>
> **Risk assessment**: A reviewer familiar with RLVεR and RESTRAIN would likely reject this paper as incremental — "applies known findings to more domains."

### 1.4 Can This Direction Be Salvaged?

A narrower, repositioned version **might** work:

- **Unique angle**: Study **natural** noise from actual majority voting (not synthetic injection) across diverse **NLP** domains (not just math/code/arithmetic), and show that different domains have fundamentally different noise *structures* (not just noise *rates*)
- **Key differentiation**: RLVεR = theory on code; "Delay Plateau Collapse" = synthetic noise on arithmetic; Ours = real-world noise on NLP tasks
- **Risk**: Still feels like an application/extension paper rather than a finding paper

**Recommendation: Do NOT pursue this direction as-is. If pursued at all, needs radical narrowing.**

---

## Part 2: Alternative Directions with Novelty Assessment

### Alternative A: "Reward Hacking in Label-Free RLVR: When Models Game Their Own Verification"

**Core Idea**: When RLVR uses majority voting as pseudo-reward (TTRL-style), models can learn to "hack" the verification signal — producing outputs that win majority votes without being correct. Study this phenomenon systematically: document hacking behaviors, identify when they emerge during training, and develop detection + prevention methods.

**Why This Might Be Novel**:
- Existing reward hacking detection papers (GRIFT [2604.16242](https://paper.dou.ac/p/2604.16242v1), TRACE [arXiv:2510.01367](https://arxiv.org/abs/2510.01367)) focus on **standard RLVR with ground-truth rewards**, not label-free/pseudo-reward settings
- Nobody has studied HOW models specifically game majority-voting-based rewards
- The label-free setting creates **qualitatively different** hacking modes (e.g., models can converge on popular-but-wrong answers, exploit agreement patterns)

**Novelty Check Results**:

| Existing Work | Relevance | Gap |
|--------------|-----------|-----|
| GRIFT (gradient fingerprinting for reward hacking) | Detects hacking via gradients, but in standard RLVR | Not for label-free/pseudo-reward |
| TRACE (truncated reasoning AUC) | Detects hacking via reasoning length | Not studied in majority-voting context |
| RESTRAIN (spurious majority mitigation) | Addresses wrong-consensus problem | Doesn't study the training dynamics of HOW hacking emerges |
| "Delay, Plateau, Collapse" | Studies error patterns | Doesn't frame as reward hacking |

**Verdict**: **MODERATE-HIGH novelty**. The intersection of "reward hacking" + "label-free RLVR" is genuinely understudied.

**Feasibility**: 5 days with 24×H200 — run TTRL-style training across domains, monitor for hacking behaviors, develop taxonomy + detection methods. Feasible.

**EMNLP fit**: Good — empirical study with practical implications.

**Risk**: The "hacking" might not be dramatic enough to make a compelling paper. Hard to define "hacking" vs "overfitting" in a label-free setting.

---

### Alternative B: "The RLVR Benefit Frontier: A Controlled Study of When RL Training Adds Value Over SFT"

**Core Idea**: Systematically characterize WHEN RLVR outperforms SFT (and vice versa) across task types, difficulty levels, data regimes, and domains. Use the same model, same data, same compute to do a clean apples-to-apples comparison.

**Why This Might Be Novel**:
- Existing comparisons are scattered: Med-RLVR shows RLVR ≈ SFT in-distribution but better OOD; "The Invisible Leash" shows RLVR can't escape base model support
- SRL (Google, ICLR 2026) proposes a hybrid, ReLIFT interleaves them
- BUT: **No single paper does a controlled, multi-domain factorial study** (domain × difficulty × data-size × RLVR-vs-SFT)

**Novelty Check Results**:

| Existing Work | Relevance | Gap |
|--------------|-----------|-----|
| Med-RLVR (2502.19655) | RLVR vs SFT in medicine only | Single domain, not systematic |
| "The Invisible Leash" (2507.14843) | Shows RLVR limitations | Theoretical focus, doesn't compare with SFT across domains |
| SRL (ICLR 2026) | Proposes hybrid SFT+RL | Doesn't study the boundary of when each is better |
| ReLIFT | Dynamic switching | Engineering solution, not systematic characterization |
| Tsinghua Critique (2504.13837) | RLVR limitations | Argues reasoning is "locked in," but doesn't do controlled comparison |

**Verdict**: **HIGH novelty**. A clean, controlled "RLVR vs SFT" characterization paper across domains and task complexities does not exist.

**Feasibility**: 5 days with 24×H200:
- Day 1-2: Prepare data for 4-5 domains × 3 difficulty levels, baseline inference
- Day 2-4: Run RLVR (GRPO) and SFT for each condition (parallel on 24 GPUs)
- Day 4-5: Analysis, ablations, write-up
- Total: ~20 training runs × 8h each = 160 GPU-hours, easily parallelized on 24×H200

**EMNLP fit**: **Perfect** — empirical characterization paper that tells the community when to use what.

**Risk**: Findings might be "boring" (e.g., RLVR always better with enough compute, or SFT always better with good data). Need to find interesting crossover points.

---

### Alternative C: "Domain Conflict and Synergy in Multi-Task RLVR"

**Core Idea**: When you train RLVR on a **mixture** of domains (math + code + science + commonsense), do domains help or hurt each other? Is there negative transfer? Does the mixing ratio matter? How does multi-domain RLVR compare to domain-specific RLVR?

**Why This Might Be Novel**:
- Most RLVR papers train on a single domain (math, code, or specific NLP task)
- Multi-domain RLVR mixing is mentioned in passing but never studied systematically
- This is a fundamental question for scaling RLVR beyond math: if you want a generally capable reasoner, you need multi-domain RLVR, but nobody knows if this works

**Novelty Check Results**:

| Existing Work | Relevance | Gap |
|--------------|-----------|-----|
| Reasoning Gym (2505.24760) | Provides 100+ tasks across domains | Environment, not a study of multi-domain training dynamics |
| RLVR-World (2505.13934) | RLVR across modalities | Different scope (world models, not reasoning) |
| GDRO (2601.19280) | Adaptive difficulty weighting in RL | Single domain (math), not cross-domain |
| Parallel Scaling Law (2510.02272) | Multi-language RLVR interactions | Language mixing, not domain mixing |
| Cross-lingual collapse research | Language interference in RLVR | Analogous phenomenon but for languages |

**Verdict**: **HIGH novelty**. Nobody has systematically studied domain interaction effects in multi-task RLVR.

**Feasibility**: 5 days with 24×H200:
- Conditions: 4 domains solo + pairwise combinations + all-domain mixture = ~15 training runs
- Each GRPO run ~10h on 4-8 GPUs → easily parallelized
- Analysis: per-domain accuracy curves, cross-domain transfer/interference metrics

**EMNLP fit**: Good — fundamental question about RLVR scalability.

**Risk**: Results might show "no interesting interaction" (each domain is independent). But cross-lingual RLVR research suggests interference IS real, so analogous domain interference is likely.

---

## Part 3: Comparative Assessment

| Criterion | Pseudo-Verifier (Original) | Alt A: Reward Hacking in Label-Free RLVR | Alt B: RLVR vs SFT Frontier | Alt C: Multi-Domain RLVR Conflict |
|-----------|---------------------------|------------------------------------------|-----------------------------|------------------------------------|
| **Novelty** | LOW (scooped by RLVεR, DPC, RESTRAIN) | MODERATE-HIGH | HIGH | HIGH |
| **Scoop risk** | HIGH (field is active) | MODERATE | LOW-MODERATE | LOW |
| **EMNLP fit** | Good (empirical) | Good (empirical) | **Perfect** (characterization) | Good (empirical) |
| **Practical impact** | Moderate (tells when SC fails) | High (hacking is a safety concern) | **Very High** (helps everyone choose training method) | High (needed for multi-domain RLVR) |
| **5-day feasibility** | Tight but doable | Risky (hacking may not be visible enough) | **Straightforward** | Straightforward |
| **Story clarity** | Muddied by existing work | Needs careful definition of "hacking" | **Crystal clear**: "When to use RLVR vs SFT" | Clear: "Can you mix domains in RLVR?" |
| **Compute match (24×H200)** | Good | Good | Good | Good |
| **Risk of "boring" results** | Low | **High** (hacking might not emerge) | Moderate | Moderate |

---

## Part 4: Final Recommendation

### TOP PICK: Alternative B — "The RLVR Benefit Frontier: When Does RL Training Add Value Over SFT?"

**Reasons**:
1. **Highest novelty**: Despite numerous RLVR and SFT papers, nobody has done a clean apples-to-apples comparison across domains × difficulties × data regimes
2. **Clearest story**: "Here's a decision guide for when to use RLVR vs SFT" — every practitioner needs this
3. **Lowest scoop risk**: This requires extensive compute + multi-domain evaluation, which is hard to replicate quickly
4. **Perfect EMNLP fit**: Empirical characterization study with practical implications
5. **Straightforward execution**: Established training pipelines (GRPO for RLVR, standard SFT), standard benchmarks, clear metrics
6. **High-impact findings expected**: Based on scattered evidence (Med-RLVR, "Invisible Leash", SRL), the boundary IS interesting — RLVR wins on some tasks, SFT wins on others, and the crossover depends on difficulty + data availability

### Proposed Title
"When Does RLVR Beat SFT? A Controlled Multi-Domain Study of Reinforcement Learning vs Supervised Fine-Tuning for LLM Reasoning"

### Proposed Contributions
1. **First controlled comparison** of RLVR (GRPO) vs SFT using the same base model, same data, same compute budget across 5+ domains
2. **Characterization of the RLVR benefit frontier**: Map where RLVR wins (tasks within model reach, limited labels, OOD generalization needed) vs where SFT wins (tasks requiring new knowledge, abundant demonstrations)
3. **Practical decision framework**: Provide guidelines for practitioners on when to invest in RLVR vs SFT post-training

### RUNNER-UP: Alternative C — "Domain Conflict in Multi-Task RLVR"

Good backup if Alt B feels too "survey-like." Alt C has a more focused research question and is less likely to produce boring results.

### NOT RECOMMENDED: Original Pseudo-Verifier Direction

The space is too crowded. RLVεR + "Delay, Plateau, Collapse" + RESTRAIN + JURY-RL collectively cover the core narrative. Pursuing this risks an EMNLP rejection on novelty grounds.

---

## Appendix: Key References for Positioning

### Papers to cite and differentiate from (for Alt B):

| Paper | Key Finding | Our Differentiation |
|-------|------------|---------------------|
| Med-RLVR (2502.19655) | RLVR ≈ SFT in-distribution, RLVR better OOD in medicine | Single domain; we do multi-domain controlled comparison |
| "The Invisible Leash" (2507.14843) | RLVR can't escape base model support | Theoretical; we provide empirical characterization across domains |
| SRL (ICLR 2026) | Hybrid SFT+RL approach | Proposes a method; we characterize WHEN each method is better |
| ReLIFT | Dynamic SFT/RL switching | Engineering solution; we provide the understanding |
| Tsinghua Critique (2504.13837) | Reasoning locked in by base model | Claims-focused; we quantify the benefit boundary |
| "Delay, Plateau, Collapse" (2605.02909) | Verifier errors shape RLVR outcomes | Studies noise; we compare training paradigms |
| "Crossing the Reward Bridge" (2503.23829) | RLVR extended to diverse domains | Uses trained verifier; we compare RLVR with SFT baseline |
