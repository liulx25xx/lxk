# Novelty Analysis: SelfCurriculum

## 1. What R-Zero Did and Didn't Do

### What R-Zero Accomplished
- **Challenger-Solver co-evolution**: Demonstrated that a single LLM can split into a task generator (Challenger) and task solver (Solver), co-evolving without any external data.
- **Uncertainty-based curriculum**: The Challenger is rewarded for generating problems where the Solver has ~50% accuracy — the sweet spot of maximum information gain.
- **GRPO-based training**: Both roles trained via Group Relative Policy Optimization with z-score normalization.
- **Majority voting as pseudo-verifier**: Correctness determined by consensus across multiple Solver samples.
- **Zero-data operation**: Started from a base LLM (Qwen3-4B/8B) with no external training data.
- **Results**: +6.49 points on math, +7.54 on general reasoning for Qwen3-4B.

### What R-Zero Did NOT Do
1. **Math-only evaluation**: All experiments are on mathematical reasoning benchmarks. No scientific, legal, or medical domains tested.
2. **No domain-aware problem generation**: The Challenger generates math problems; no mechanism to steer generation toward specific knowledge domains.
3. **Single pseudo-verifier strategy**: Relies solely on majority voting from the Solver. No cross-model agreement, no process-level verification.
4. **No quality control for pseudo-labels**: The majority vote can be systematically wrong for domains where the model has consistent blind spots.
5. **No domain-specific verification signals**: For non-math domains (science, law), answer correctness isn't easily determined by string matching — needs semantic understanding.
6. **No multi-domain training**: Cannot handle a curriculum spanning multiple domains simultaneously.
7. **No analysis of pseudo-verifier reliability**: No study of when majority voting fails and how to detect/mitigate it.

---

## 2. Gaps in the Broader Literature

### Gap 1: Self-Evolving Curriculum Has Not Been Applied Beyond Math/Code
- **R-Zero**: Math only
- **SQLM**: Math + coding (with unit tests for coding)
- **OpenSIR**: Math only
- **WebRL**: Web navigation (not reasoning)
- **SPIRAL**: Game-playing → math transfer
- **Status**: No self-evolving curriculum for science, law, or medicine reasoning.

### Gap 2: Pseudo-Verification Is Primitive and Unstudied
- Current approaches use **majority voting** (TTRL, R-Zero, SQLM).
- TTRL shows 92% reward accuracy vs 16% label accuracy — but only for math where answers are well-defined.
- For open-ended domains (legal reasoning, scientific explanation), majority voting degrades:
  - Models may consistently agree on incorrect domain-specific answers.
  - Semantic equivalence of answers is hard to determine by string matching.
- **RLCCF** uses multi-model voting but focuses on math benchmarks.
- **"Crossing the Reward Bridge"** trains a 7B verifier — requires labeled data and domain-specific training.
- **No work combines** self-consistency + cross-model agreement as a training-free pseudo-verifier for arbitrary domains.

### Gap 3: Curriculum Methods Are Passive, Not Generative
- **VCRL**: Selects from fixed dataset based on reward variance.
- **DOTS**: Selects from fixed dataset based on estimated difficulty.
- **AdaCuRL**: Partitions fixed dataset into difficulty buckets.
- **E2H Reasoner**: Schedules fixed tasks from easy to hard.
- **None of these generate new problems** — they only reorder existing ones.
- Self-evolving methods (R-Zero, SQLM) generate problems but lack curriculum-aware difficulty control across domains.

### Gap 4: No Unified Multi-Domain Self-Evolving Framework
- Domain-specific RLVR works are isolated:
  - Med-RLVR for medicine
  - WildSci for science
  - Genome-Bench for genomics
  - LegalBench for law
- Each requires its own reward design, data pipeline, and training recipe.
- No framework that can instantiate self-evolving curricula across domains with a common architecture.

### Gap 5: Verification Reliability Is Unquantified
- TTRL's "lucky hit" analysis only covers math.
- No systematic study of pseudo-verifier accuracy across domains.
- No adaptive mechanism to increase verification reliability when the model is uncertain.
- PRISM combines PRM + self-certainty, but requires a trained PRM — not training-free.

---

## 3. Our Unique Contributions

### Contribution 1: Domain-Agnostic Self-Evolving Curriculum (SelfCurriculum)
**What's new**: A unified framework that extends R-Zero's Challenger-Solver paradigm to arbitrary reasoning domains (science, law, medicine) through domain-conditioned problem generation and domain-aware pseudo-verification.

**Why it matters**: R-Zero showed self-evolution works for math; we show it can work for any domain with multiple-choice or short-answer structure — dramatically expanding the applicability of autonomous LLM self-improvement.

### Contribution 2: Composite Pseudo-Verifier (CPV)
**What's new**: A training-free verification mechanism combining three signals:
1. **Self-Consistency (SC)**: Multiple samples from the same model; agreement = confidence.
2. **Cross-Model Agreement (CMA)**: Agreement between the training model and a held-out reference model.
3. **Confidence-Weighted Fusion**: Dynamically adjusts the weight of SC vs. CMA based on domain-specific reliability estimates.

**Why it matters**: Unlike "Crossing the Reward Bridge" (requires training a 7B verifier), our CPV is training-free and domain-agnostic. Unlike pure majority voting (TTRL, R-Zero), CPV uses cross-model signals to catch systematic model-specific blind spots.

### Contribution 3: Domain-Conditioned Adaptive Curriculum
**What's new**: The Challenger generates domain-specific problems with explicit difficulty control:
- Problems are generated with domain tags and difficulty targets.
- Difficulty is measured by the Solver's success rate on that problem type.
- The curriculum adapts: starts easy, ramps difficulty as the Solver improves, per domain.
- Cross-domain transfer: difficulty progression is tracked independently per domain but shared training updates benefit all domains.

**Why it matters**: Existing curriculum methods (VCRL, DOTS) select from fixed pools; existing self-play methods (R-Zero) don't have fine-grained domain or difficulty control. We combine self-generation with adaptive curriculum scheduling.

### Contribution 4: Systematic Study of Pseudo-Verifier Reliability Across Domains
**What's new**: First empirical study comparing pseudo-verifier accuracy across diverse domains:
- When does self-consistency work? (High for math, moderate for MCQ science, low for open-ended legal reasoning)
- When does cross-model agreement help? (Most valuable when models have complementary knowledge)
- How does reliability degrade and what are mitigation strategies?

**Why it matters**: Critical for understanding the practical limits of training-free RLVR beyond math.

### Contribution 5: Multi-Domain Evaluation at Scale
**What's new**: Evaluate self-evolving curriculum across 5+ diverse reasoning domains:
- Science: ScienceQA, SciBench, ARC-Challenge
- Law: LegalBench
- Medicine: MedQA, PubMedQA
- General: MMLU-Pro subsets
- Math (for comparison): MATH, GSM8K

**Why it matters**: First comprehensive cross-domain evaluation of self-evolving curriculum RL, enabling direct comparison with domain-specific methods.

---

## 4. Positioning Against Closest Competitors

| Feature | R-Zero | SQLM | VCRL | Crossing the Reward Bridge | **SelfCurriculum (Ours)** |
|---------|--------|------|------|---------------------------|--------------------------|
| Self-generated problems | Yes | Yes | No | No | **Yes** |
| Domain-agnostic | No (math) | No (math/code) | No (math) | Yes (but needs trained verifier) | **Yes (training-free)** |
| Adaptive curriculum | Implicit (uncertainty) | Implicit | Yes (variance) | No | **Yes (explicit + adaptive)** |
| Pseudo-verifier | Majority vote | Majority vote | N/A | Trained 7B model | **CPV: SC + CMA fusion** |
| Cross-model signals | No | No | No | No | **Yes** |
| Multi-domain evaluation | No | No | No | Limited | **Comprehensive (5+ domains)** |
| No external labeled data | Yes | Yes | Yes (needs dataset) | No (needs 160K labels) | **Yes** |

---

## 5. Potential Weaknesses and Mitigations

### Weakness 1: "Pseudo-verifier might not work well for truly open-ended tasks"
**Mitigation**: We focus on tasks with verifiable answers (MCQ, short-answer). For each domain, we carefully select benchmarks with clear correct answers. We also provide an honest analysis of when CPV degrades.

### Weakness 2: "Generated problems might not be domain-faithful"
**Mitigation**: Domain-conditioned generation with few-shot exemplars; quality filtering via CPV itself (discard problems where CPV gives uncertain verification scores).

### Weakness 3: "Cross-model agreement requires an additional model"
**Mitigation**: The reference model can be small (1.5B–7B); inference cost is amortized over many training iterations; we show the cost-benefit tradeoff.

### Weakness 4: "Not truly 'zero data' — needs domain exemplars"
**Mitigation**: We use only a small seed set (100–500 exemplars per domain) from publicly available benchmarks, which is realistic and far less than training a domain verifier. We also ablate seed set size.
