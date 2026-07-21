# Paper Outline: SelfCurriculum

## Title
**SelfCurriculum: Self-Evolving Curriculum Learning for Domain-Specific LLM Reasoning**

## Target
EMNLP 2026 (Main Conference, Long Paper, 8 pages + references)

---

## Abstract (~250 words)

**[Motivation]** Recent advances in reinforcement learning with verifiable rewards (RLVR) have dramatically improved LLM reasoning, with methods like DeepSeek-R1 and GRPO achieving remarkable results on mathematical benchmarks. However, these successes rely on domains with natural ground-truth verifiers (math checkers, code executors). Extending RLVR to knowledge-intensive domains — science, law, medicine — where ground-truth verification is unavailable remains an open challenge.

**[Gap]** Self-evolving approaches like R-Zero demonstrate that LLMs can autonomously generate training curricula through Challenger-Solver co-evolution, but are limited to math and rely on simple majority voting for verification. For non-math domains, majority voting is unreliable due to systematic model biases.

**[Method]** We propose SelfCurriculum, a unified framework for autonomous LLM self-improvement across diverse reasoning domains. SelfCurriculum introduces: (1) a domain-conditioned Challenger that generates reasoning problems with adaptive difficulty; (2) a Composite Pseudo-Verifier (CPV) combining self-consistency with cross-model agreement for training-free reward signals; and (3) an adaptive curriculum controller that tracks and optimizes difficulty progression independently per domain.

**[Results]** Experiments across science (ScienceQA, ARC-Challenge), law (LegalBench), medicine (MedQA), and math benchmarks show that SelfCurriculum outperforms GRPO baselines by X-Y% across non-math domains while matching R-Zero on math. We demonstrate that CPV provides Z% more accurate rewards than majority voting for knowledge-intensive domains, and that domain-adaptive curriculum scheduling yields consistent improvements over static training.

**[Contribution]** To our knowledge, SelfCurriculum is the first framework to achieve autonomous self-improving LLM training across diverse knowledge domains without external verifiers or labeled data.

---

## 1. Introduction (~1.5 pages)

### Paragraph 1: The RLVR Revolution
- RLVR has transformed LLM reasoning: DeepSeek-R1, GRPO, DAPO achieve near-human math performance.
- Key enabler: verifiable rewards — math has automatic checkers, code has test suites.
- Cite: DeepSeek-R1 (2501.12948), DeepSeekMath/GRPO (2402.03300), DAPO (2503.14476).

### Paragraph 2: The Domain Gap
- Real-world reasoning extends far beyond math: science, law, medicine, finance.
- These domains lack natural verifiers — no equivalent of "run the code."
- Current extensions (Crossing the Reward Bridge) require training domain-specific verifier models — expensive and not scalable.
- Cite: Crossing the Reward Bridge (2503.23829), Med-RLVR (2502.19655), WildSci (2601.05567).

### Paragraph 3: Self-Evolving Curriculum — Promise and Limits
- R-Zero showed LLMs can self-generate curricula: Challenger creates problems, Solver learns from them.
- But: (a) math-only, (b) majority-vote verification is unreliable for knowledge domains, (c) no domain-aware difficulty control.
- Cite: R-Zero (2508.05004), SQLM (2508.03682), OpenSIR (2511.00602).

### Paragraph 4: Our Approach — SelfCurriculum
- Three key innovations: (1) domain-conditioned generation, (2) composite pseudo-verifier, (3) adaptive curriculum.
- Training-free verification: no labeled data, no trained verifier models.
- Unified framework instantiable across science, law, medicine.
- Brief overview of results.

### Paragraph 5: Contributions
1. First domain-agnostic self-evolving curriculum framework for LLM reasoning.
2. Composite Pseudo-Verifier combining self-consistency + cross-model agreement.
3. Adaptive curriculum controller with per-domain difficulty tracking.
4. Comprehensive evaluation across 5+ reasoning domains showing consistent improvements.
5. Systematic study of pseudo-verifier reliability across knowledge domains.

---

## 2. Related Work (~1.5 pages)

### 2.1 Reinforcement Learning with Verifiable Rewards
- DeepSeek-R1, GRPO, DAPO, STILL-3: Math/code reasoning via RL.
- RLVR extensions: Crossing the Reward Bridge (trained verifier), Med-RLVR (medical MCQ), WildSci (science), Genome-Bench (genomics).
- Gap: All require either ground-truth or a trained domain verifier.

### 2.2 Self-Evolving and Self-Play Methods
- R-Zero: Challenger-Solver co-evolution (math only).
- SQLM: Proposer-Solver asymmetric self-play.
- OpenSIR: Teacher-Student self-play with diversity optimization.
- SPIRAL: Zero-sum game self-play for reasoning transfer.
- WebRL: Self-evolving curriculum for web agents.
- Gap: None extend to knowledge-intensive non-math domains.

### 2.3 Curriculum Learning for LLM RL
- VCRL: Variance-based difficulty selection (math).
- DOTS: Optimal difficulty at 50% success rate.
- AdaCuRL: Coarse-to-fine difficulty with invalid sample mitigation.
- E2H Reasoner: Easy-to-hard with convergence guarantees.
- Learning Like Humans: ADCL + EGSR (EMNLP 2025).
- MRACL: Multi-reward adaptive curriculum (AAAI 2026).
- Gap: All select from fixed problem pools; none generate problems.

### 2.4 Pseudo-Verification and Self-Consistency
- TTRL: Majority-vote pseudo-labels for test-time RL (math).
- RLCCF: Multi-model collective feedback (math).
- Self-Rewarding LMs: LLM-as-a-Judge iterative DPO.
- PRISM: PRM + self-certainty for unlabeled training.
- PRIME: Implicit process rewards from outcome labels.
- Gap: No composite approach combining SC + CMA; no cross-domain reliability study.

---

## 3. Method: SelfCurriculum (~2.5 pages)

### 3.1 Problem Formulation
- Self-evolving RL loop: At iteration $t$, Challenger $\pi_\phi$ generates problems, Solver $\pi_\theta$ solves them, CPV provides rewards, GRPO updates both.
- Domains $\mathcal{D} = \{d_1, ..., d_K\}$ with seed exemplars $\mathcal{E}_d$.
- No external labels or trained verifiers required.

### 3.2 Domain-Conditioned Challenger
- Prompt template with domain tag, difficulty target, format.
- Challenger reward: $R_C = \alpha \cdot r_{unc} + \beta \cdot r_{domain} + \gamma \cdot r_{div}$
  - Uncertainty reward (target 50% solver accuracy)
  - Domain faithfulness reward
  - Diversity reward (embedding distance)
- Trained via GRPO jointly with Solver.

### 3.3 Composite Pseudo-Verifier (CPV)
- **Self-Consistency (SC)**: $K$ rollouts from Solver; majority answer confidence.
- **Cross-Model Agreement (CMA)**: $M$ rollouts from independent reference model.
- **Confidence-Weighted Fusion**: $CPV = w_d \cdot SC + (1 - w_d) \cdot CMA$
  - $w_d$ estimated per domain on small validation set.
- Analysis of when SC vs. CMA is more reliable.
- [Figure 2: CPV architecture diagram]

### 3.4 Adaptive Curriculum Controller
- Per-domain difficulty tracking: $\delta_d^{(t)} = 1 - \text{mean accuracy}_d$.
- Adaptive scheduling: increase difficulty if accuracy > 0.6, decrease if < 0.4.
- Domain mixing: proportional to learning opportunity gap.
- Memory bank for replay of high-value problems.
- [Figure 3: Curriculum progression visualization]

### 3.5 Training Algorithm
- Full algorithm box (Algorithm 1).
- GRPO details: group size, clipping, KL penalty.
- Computational complexity analysis.

---

## 4. Experimental Setup (~1 page)

### 4.1 Datasets
- Table listing all benchmarks: ScienceQA, ARC-Challenge, SciBench, LegalBench, MedQA, MMLU-Pro, MATH, GSM8K.
- Seed exemplar sizes per domain.

### 4.2 Baselines
- Base model, SFT, GRPO variants, R-Zero, DPO baselines.
- Fair comparison protocol: same compute budget, same base model.

### 4.3 Implementation Details
- Model: Qwen2.5-7B (primary), Qwen2.5-14B (scaling).
- Reference: Llama-3.1-8B-Instruct.
- Training: veRL framework, 5 iterations, 16 rollouts, lr=1e-6.
- Hardware: 24x H200 GPUs.

### 4.4 Evaluation Protocol
- Pass@1 accuracy, 0-shot.
- CoT prompting for all methods.
- Statistical significance: 3 seeds, report mean ± std.

---

## 5. Results (~2 pages)

### 5.1 Main Results (Table 1)
- SelfCurriculum vs. all baselines across all benchmarks.
- [Table 1: Main results — the key table of the paper]
- Highlight: consistent improvement on non-math domains.
- Matches/approaches R-Zero on math without math-specific design.

### 5.2 CPV Reliability Analysis (Table 2 + Figure)
- [Table 2: Pseudo-verifier accuracy per domain]
- SC accuracy, CMA accuracy, CPV accuracy, SC-CMA agreement rate.
- [Figure 4: CPV accuracy vs. domain complexity scatter plot]
- Key finding: CMA most valuable for domains where model has blind spots.

### 5.3 Curriculum Progression (Figure)
- [Figure 5: Difficulty progression across iterations, per domain]
- Show adaptive difficulty tracks with solver capability.
- Convergence behavior: science plateaus earlier than law.

### 5.4 Ablation Studies
- [Table 3: CPV component ablation]
- [Table 4: Curriculum strategy ablation]
- [Table 5: Iteration count vs. performance]
- Key findings: Full CPV > SC-only > Majority Vote; Adaptive curriculum > Fixed > Random.

### 5.5 Scaling Analysis
- [Figure 6: Performance vs. model size (1.5B, 7B, 14B)]
- Larger models benefit more (consistent with TTRL findings).
- CPV accuracy improves with model quality.

---

## 6. Analysis (~1 page)

### 6.1 What Kinds of Reasoning Improve?
- Breakdown by reasoning type: factual recall, multi-step inference, rule application, causal reasoning.
- SelfCurriculum improves multi-step and rule application most; factual recall less.

### 6.2 Cross-Domain Transfer
- [Table 6: Transfer matrix]
- Joint multi-domain training shows positive transfer (science ↔ medicine).
- Law shows least transfer (most domain-specific).

### 6.3 Generated Problem Quality
- Human evaluation of 100 generated problems per domain.
- Validity, domain faithfulness, difficulty calibration.
- Quality improves with iteration (Challenger co-evolves).

### 6.4 Failure Mode Analysis
- When does SelfCurriculum fail?
- Main failure mode: CPV gives confident wrong reward for domain-specific knowledge.
- Frequency: ~5% for science MCQ, ~12% for legal reasoning.

### 6.5 Case Studies
- [Figure 7: Example problems and solutions from different domains]
- Show Challenger generates increasingly sophisticated problems.
- Show Solver develops domain-specific reasoning patterns.

---

## 7. Discussion (~0.5 pages)

### 7.1 When Does SelfCurriculum Work Best?
- Domains with verifiable-format answers (MCQ, short-answer).
- Models with sufficient base capability (7B+).
- When cross-model agreement adds value (complementary knowledge).

### 7.2 Limitations
- Requires MCQ or verifiable answer format (not free-form).
- Needs a different-family reference model for CMA.
- Pseudo-verifier accuracy still below oracle (ground-truth) rewards.
- Generated problems may not cover all domain subtopics.

### 7.3 Broader Impact
- Enables autonomous domain-specific LLM improvement without expert annotation.
- Reduces cost barrier for specialized AI training.
- Potential risks: model confidently learns incorrect domain knowledge.

---

## 8. Conclusion (~0.25 pages)

- Summary of contributions.
- SelfCurriculum: first domain-agnostic self-evolving curriculum for LLM reasoning.
- CPV enables training-free verification across diverse domains.
- Consistent improvements on science, law, medicine benchmarks.
- Future work: extend to open-ended generation tasks; improve CPV for ambiguous domains; combine with retrieval augmentation.

---

## References (~1-2 pages, not counted in page limit)

Key references (32 papers):
1. DeepSeek-R1 (2501.12948)
2. DeepSeekMath/GRPO (2402.03300)
3. DAPO (2503.14476)
4. R-Zero (2508.05004)
5. SQLM (2508.03682)
6. OpenSIR (2511.00602)
7. SPIRAL (2506.24119)
8. WebRL (2411.02337)
9. VCRL (2509.19803)
10. AdaCuRL (2511.09478)
11. DOTS (2506.05316)
12. E2H Reasoner (2506.06632)
13. TTRL (2504.16084)
14. RLCCF (2508.12338)
15. Self-Rewarding LMs (2401.10020)
16. PRISM (2601.04700)
17. PRIME (2502.01456)
18. Crossing the Reward Bridge (2503.23829)
19. WildSci (2601.05567)
20. Med-RLVR (2502.19655)
21. Genome-Bench (2505.19501)
22. STILL-3 (2503.04548)
23. Learning Like Humans (2505.08364)
24. MRACL (AAAI 2026)
25. Multi-Agent Evolve (2510.23595)
26. RLVE (2511.07317)
27. Meta-Rewarding (2407.19594)
28. MMLU-Pro (2406.01574)
29. SciBench (2307.10635)
30. LegalBench (NeurIPS 2024)
31. ScienceQA (2209.09513)
32. MedQA (2009.13081)

---

## Appendix

### A. Full Algorithm Pseudocode
### B. Domain Prompt Templates
### C. Hyperparameter Sensitivity Analysis
### D. Additional Ablation Tables
### E. Generated Problem Examples (Full)
### F. Human Evaluation Protocol and Inter-Annotator Agreement
### G. Compute Details and Reproducibility

---

## Figures and Tables Summary

| ID | Type | Content | Section |
|----|------|---------|---------|
| Figure 1 | Architecture | SelfCurriculum overview diagram | §3.1 |
| Figure 2 | Architecture | CPV design | §3.3 |
| Figure 3 | Visualization | Curriculum progression (difficulty vs. iteration) | §3.4 |
| Figure 4 | Analysis | CPV accuracy vs. domain complexity | §5.2 |
| Figure 5 | Visualization | Per-domain difficulty trajectory | §5.3 |
| Figure 6 | Scaling | Performance vs. model size | §5.5 |
| Figure 7 | Case Study | Example problems and solutions | §6.5 |
| Table 1 | Results | Main results (all benchmarks) | §5.1 |
| Table 2 | Analysis | CPV reliability per domain | §5.2 |
| Table 3 | Ablation | CPV component ablation | §5.4 |
| Table 4 | Ablation | Curriculum strategy ablation | §5.4 |
| Table 5 | Ablation | Iteration count | §5.4 |
| Table 6 | Analysis | Cross-domain transfer matrix | §6.2 |
