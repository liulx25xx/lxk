# Comprehensive Literature Survey: Self-Evolving Curriculum Learning for Domain-Specific LLM Reasoning

## 1. Reinforcement Learning with Verifiable Rewards (RLVR) for LLM Reasoning

### 1.1 Foundational RLVR: DeepSeek-R1 and GRPO

**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**
- arXiv: 2402.03300 (Shao et al., 2024)
- Introduced **Group Relative Policy Optimization (GRPO)**, a variant of PPO that eliminates the critic network by estimating advantages using group-based reward normalization.
- Key innovation: Replaces PPO's value function with group-based reward comparison — rewards are normalized within a group of sampled responses (z-score normalization).
- Achieved 51.7% on MATH benchmark (60.9% with self-consistency), rivaling GPT-4.
- GRPO reduces memory overhead by ~30% compared to PPO while maintaining training stability.
- Advantage formula: $\hat{A}_i = \frac{r_i - \mu(r_1,...,r_G)}{\sigma(r_1,...,r_G)}$

**DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
- arXiv: 2501.12948 (DeepSeek-AI, 2025)
- Full pipeline: SFT → GRPO RL → Rejection Sampling → Final RL alignment.
- Multi-dimensional rewards: correctness (binary), format adherence, cosine similarity to reference solutions.
- Achieved 97.3% on MATH, 79.8% on AIME 2024.
- Emergent behaviors: self-correction, extended reasoning chains, "aha moments."

### 1.2 GRPO Improvements

**DAPO: An Open-Source LLM Reinforcement Learning System at Scale**
- arXiv: 2503.14476 (2025)
- Key innovations over GRPO:
  1. **Decoupled Clipping**: Asymmetric bounds (ε_high for positive, ε_low for negative advantages) to encourage exploration.
  2. **Dynamic Sampling**: Filters uninformative prompts (all correct or all incorrect), improving sample efficiency by ~50%.
  3. **Token-Level Loss**: Replaces sample-level with token-level granularity for better credit assignment.
  4. **Overlong Reward Shaping**: Smooth length-based penalties.
- Achieved 50/32 on AIME 2024 with Qwen2.5-32B, surpassing DeepSeek-R1.

**STILL-3: An Empirical Study on Eliciting and Improving R1-like Reasoning Models**
- arXiv: 2503.04548 (2025)
- On-policy online RL for reasoning activation.
- STILL-3-Zero-32B: improved from 2.08% → 37.08% on AIME 2024.
- STILL-3-Tool-32B: 86.67% on AIME 2024 with Python code execution integration.
- Key finding: Minimal data efficiency — RL effective even with as few as 4 problems.

---

## 2. Self-Evolving and Curriculum Reinforcement Learning

### 2.1 R-Zero: The Direct Predecessor

**R-Zero: Self-Evolving Reasoning LLM from Zero Data**
- arXiv: 2508.05004 (Huang et al., Tencent AI Seattle Lab, 2025)
- **Core Architecture**: Challenger-Solver co-evolution from a single base LLM.
  - **Challenger** (Task Generator): Rewarded for generating tasks near the Solver's capability boundary (~50% accuracy = maximum uncertainty).
  - **Solver** (Task Executor): Rewarded for correctly solving tasks; correctness via majority-vote pseudo-labels.
- **Uncertainty Reward**: $r_{uncertainty} = 1 - 2|p̂(x; S_φ) - 0.5|$
- **Training**: GRPO with z-score-normalized advantages and KL-divergence regularization.
- **Results**: Qwen3-4B: +6.49 points on math, +7.54 on general reasoning; Qwen3-8B: +5.51 points.
- **Limitation**: Focused exclusively on math; no domain adaptation mechanism; relies on majority voting which works well for math but is unreliable for open-ended domains.

### 2.2 Self-Questioning Language Models (SQLM)

**Self-Questioning Language Models**
- arXiv: 2508.03682 (2025)
- **Asymmetric self-play**: Proposer generates questions, Solver answers them, both trained via RL.
- **Proposer reward**: Based on generating problems neither too easy nor too hard for the solver.
- **Solver reward**: Majority voting or unit tests as correctness proxies.
- **Results**: +16% on arithmetic/algebra; +7% on coding tasks (Codeforces) — without external data.
- **Limitations**: Relies on heuristic rewards (majority voting); open-ended tasks remain challenging.

### 2.3 OpenSIR: Open-Ended Self-Improving Reasoner

**OpenSIR: Open-Ended Self-Improving Reasoner**
- arXiv: 2511.00602 (EdinburghNLP, 2025)
- Self-play framework: "Teacher" (problem generator) and "Student" (solver) roles.
- Optimizes for difficulty + diversity via dual-dimensional novelty scoring.
- Can bootstrap from a single trivial seed problem.
- Results: Gemma-2-2B: GSM8K 38.5→58.7; Llama-3.2-3B: GSM8K 73.9→78.3.
- Uses embedding distance to measure conceptual diversity.

### 2.4 SPIRAL: Self-Play on Zero-Sum Games

**SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning**
- arXiv: 2506.24119 (2025)
- Multi-agent RL through competitive self-play in zero-sum games (TicTacToe, Kuhn Poker).
- **Role-Conditioned Advantage Estimation (RAE)**: Maintains separate advantage estimates per player role.
- Training on games improves math/general reasoning by up to 10%.
- Even DeepSeek-R1-Distill-Qwen-7B shows 2% improvement after SPIRAL.
- Generates automatic curriculum without human supervision.

### 2.5 Multi-Agent Evolve (MAE)

**Multi-Agent Evolve: LLM Self-Improve through Co-evolution**
- arXiv: 2510.23595 (2025)
- Three agents: **Proposer**, **Solver**, **Judge** — all from a single LLM.
- Closed-loop: Proposer generates tasks → Solver attempts → Judge evaluates both question quality and answer quality.
- 4.54% average improvement on benchmarks (Qwen2.5-3B-Instruct).
- Limitation: Depends on high-quality Judge; limited question diversity.

---

## 3. Curriculum Learning for LLM Reinforcement Learning

### 3.1 WebRL: Self-Evolving Online Curriculum RL

**WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning**
- ICLR 2025 (Qi et al., 2025); arXiv: 2411.02337
- Self-evolving curriculum for web agent training.
- Dynamic task difficulty adjustment based on agent performance.
- Combines online RL with curriculum scheduling.
- Demonstrates self-evolving curriculum can work beyond math — in web navigation.

### 3.2 VCRL: Variance-Based Curriculum RL

**VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models**
- arXiv: 2509.19803 (2025)
- **Key insight**: Variance of rewards in a rollout group reflects sample difficulty.
  - Too easy/hard → low variance; Moderate difficulty → high variance = maximum learning signal.
- **Dynamic Sampling**: Prioritizes high-variance (moderate difficulty) samples.
- **Replay Memory Bank**: Priority queue stores high-value samples for replay.
- Results: +4.67 average points on Qwen3-8B over GRPO/DAPO baselines.
- **Math-only evaluation**: AIME-2024, MATH500 benchmarks.

### 3.3 AdaCuRL: Adaptive Curriculum RL

**AdaCuRL: Adaptive Curriculum Reinforcement Learning with Invalid Sample Mitigation and Historical Revisiting**
- arXiv: 2511.09478 (Alibaba, 2025)
- Problems addressed: Gradient starvation and policy degradation.
- **Coarse-to-Fine Difficulty Estimation**: Categorize → refined score $d(q) = 1 - c(q)/N$.
- **Adaptive Scheduling**: Data partitioned into difficulty buckets; merging based on capability score.
- **Invalid Sample Mitigation**: Sparse KL divergence + reference model reset.
- **Historical Revisiting**: Re-estimates difficulty, discards mastered samples, retrains.
- Outperforms GRPO on math and general reasoning.

### 3.4 DOTS: Difficulty-Targeted Online Data Selection

**Improving Data Efficiency for LLM RL Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay**
- arXiv: 2506.05316 (2025)
- **Optimal difficulty at 0.5**: Maximizes expected squared gradient norm (proven via Theorem 1).
- **Attention-based prediction**: Estimates difficulty via similarity-weighted attention over reference set embeddings.
- Sampling probability: $\propto \exp(-|D̂_t(q) - 0.5| / τ)$
- Results: 23–65% reduction in training time while matching baseline performance.
- Pearson correlation > 0.7 for difficulty prediction.

### 3.5 E2H Reasoner: Easy to Hard Curriculum

**Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning**
- arXiv: 2506.06632 (2025)
- Theoretical convergence guarantees within approximate policy iteration framework.
- Reduces total training samples by up to 40% vs. non-curriculum RL.
- Three stages: Bootstrapping → Multitasking → Adversarial RL.
- Limitation: Manual curriculum design may be needed for new domains.

### 3.6 MRACL: Multi-Reward Space Guided Adaptive Curriculum RL

**MRACL: Multi-Reward Space Guided Adaptive Curriculum Reinforcement Learning for LLMs**
- AAAI 2026 (Liu et al., 2026)
- Multiple reward signals guide adaptive curriculum scheduling.
- Combines multi-objective optimization with adaptive task scheduling.

### 3.7 Learning Like Humans (EMNLP 2025)

**Learning Like Humans: Advancing LLM Reasoning via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation**
- EMNLP 2025 (arXiv: 2505.08364)
- **ADCL**: Addresses "Difficulty Shift" — dynamically re-estimates difficulty during training.
- **Normalized Inversion Rate (NIR)** metric to quantify re-sorting effectiveness.
- **EGSR**: Guides model to reformulate expert solutions in its own conceptual framework.
- Results: +10% over Zero-RL on AIME24; +16.6% on AIME25.

---

## 4. RLVR Beyond Math: Domain-Specific Extensions

### 4.1 Crossing the Reward Bridge

**Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains**
- arXiv: 2503.23829 (2025)
- **Generative Verifier**: 7B-parameter reward model (RM-7B) fine-tuned on 160K samples.
- Extends RLVR to medicine, economics, psychology, chemistry.
- **Soft rewards** outperform binary rewards for unstructured tasks.
- $\tilde{r}(x, a, y_i) = \frac{r(x, a, y_i) - \mu_r}{\sigma_r}$ (z-score normalized token-level probabilities)
- 7B reward model matches performance of 72B models.
- Cross-model agreement: Cohen's κ > 0.86–0.88 with expert references.

### 4.2 WildSci: Scientific Reasoning from Literature

**WildSci: Advancing Scientific Reasoning from In-the-Wild Literature**
- arXiv: 2601.05567 (2025)
- 9 scientific disciplines, 26 subdomains.
- MCQ format → scalable RL training with clear reward signals.
- Generated from real peer-reviewed literature.
- Demonstrates RL-based training dynamics for scientific reasoning.
- Dataset on HuggingFace: JustinTX/WildSci.

### 4.3 Genome-Bench: Scientific Reasoning for Genomics

**Genome-Bench**
- arXiv: 2505.19501 (2025)
- 3,332 MCQs from 11 years of CRISPR gene-editing forum discussions.
- Real-world scientific reasoning: troubleshooting lab protocols, reagent selection.
- Automated pipeline: forum threads → RL-compatible Q&A pairs via LLM extraction.
- Demonstrates RLVR training elicits spontaneous multi-step reasoning in genomics.

### 4.4 Med-RLVR: Medical Domain RL

**Med-RLVR: Reinforcement Learning with Verifiable Rewards for Medical Reasoning**
- arXiv: 2502.19655 (2025)
- 3B-parameter model trained solely on verifiable MCQ labels.
- Achieves parity with SFT and 8% boost in out-of-distribution generalization.
- Uses deterministic, rule-based rewards (format + answer correctness).
- Demonstrates emergent reasoning from simple reward signals.

### 4.5 RLVE: Verifiable Environments

**RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments**
- arXiv: 2511.07317 (2025)
- 400 procedurally generated verifiable environments spanning programming, symbolic math, NP-complete problems.
- **Adaptive difficulty window** [ℓπ, hπ] slides upward as model improves.
- 3.37% average improvement across six reasoning benchmarks using 3× less compute.
- Outperforms static datasets (DeepMath-103K) by ~2%.
- Pedagogical principles: teaches reasoning processes, not just solutions.

---

## 5. Pseudo-Verification and Self-Consistency Approaches

### 5.1 TTRL: Test-Time Reinforcement Learning

**TTRL: Test-Time Reinforcement Learning**
- arXiv: 2504.16084 (PRIME-RL, 2025)
- **Unsupervised self-improvement during inference** via majority voting pseudo-labels.
- Generates N diverse outputs → most frequent answer = consensus label y* → binary reward.
- On AIME 2024: improved Qwen-2.5-Math-7B by 159% — without labeled data.
- **"Lucky Hit" Phenomenon**: Even incorrect pseudo-labels yield correct negative rewards (92% reward accuracy vs. 16% label accuracy).
- Trains using GRPO or PPO optimization on pseudo-labeled data.

### 5.2 RLCCF: Coevolutionary Collective Feedback

**Wisdom of the Crowd: Reinforcement Learning from Coevolutionary Collective Feedback**
- arXiv: 2508.12338 (Fudan University, 2025)
- Multi-model voting with Self-Consistency (SC)-weighted rewards.
- Optimizes **Collective Consistency (CC)** across diverse LLM ensemble.
- 16.72% average relative accuracy improvement for individual models.
- 4.51% boost in majority-voting accuracy for the collective.
- Enables collaborative evolution without external supervision.

### 5.3 Self-Rewarding Language Models

**Self-Rewarding Language Models**
- arXiv: 2401.10020 (Yuan et al., Meta, 2024)
- Model serves dual roles: instruction following + self-evaluation (LLM-as-a-Judge).
- Iterative DPO: Generate → Self-Score → Create Preference Pairs → DPO Training.
- After 3 iterations, outperformed Claude 2, Gemini Pro, GPT-4-0613 on AlpacaEval 2.0.
- Follow-up: **Meta-Rewarding** (arXiv: 2407.19594) — LLM-as-a-Meta-Judge improves win rate 22.9%→39.4%.

### 5.4 PRISM: Post-Training Without Verifiable Rewards

**PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards**
- arXiv: 2601.04700 (Arizona State + AWS, 2025)
- Combines **Process Reward Model (PRM)** with model's internal **self-certainty**.
- Evaluates intermediate reasoning steps, not just final outputs.
- Avoids dependency on ground-truth labels; stabilizes training with unlabeled data.
- Optimized for math reasoning and code generation.

### 5.5 PRIME: Process Reinforcement through Implicit Rewards

**PRIME: Process Reinforcement through Implicit Rewards**
- arXiv: 2502.01456 (2025)
- Trains outcome reward model using only final-result labels, repurposes as PRM.
- Dense token-level rewards mitigate sparsity.
- 15.1–16.7% average improvement over SFT with 10% of training data.
- Online updates reduce distribution shifts.

---

## 6. Relevant Benchmarks

### 6.1 SciBench
- ICML 2024 (arXiv: 2307.10635)
- College-level scientific problems: math, chemistry, physics.
- Best LLMs achieve only 43.22% accuracy.
- Tests 10 core problem-solving abilities.

### 6.2 MMLU-Pro
- NeurIPS 2024 (arXiv: 2406.01574)
- 12,000+ curated questions across 14 domains.
- 10 answer choices (up from 4 in MMLU).
- GPT-4o: 72.55%; models show 16–33% lower accuracy vs MMLU.

### 6.3 LegalBench
- NeurIPS 2024
- 162 tasks spanning 6 types of legal reasoning.
- Evaluates issue-spotting, rule recall/application, multi-hop reasoning.
- GPT-4 outperforms specialized legal models in zero-shot settings.

### 6.4 ARC-Challenge
- AI2 Reasoning Challenge: 7,787 science questions.
- Recent study (arXiv: 2412.17758) shows difficulty partly stems from evaluation setup.
- When models evaluate all options together: 64%→93% accuracy.

---

## 7. Summary of Key Gaps Identified

| Gap | Current State | Our Opportunity |
|-----|---------------|-----------------|
| R-Zero is math-only | No domain adaptation mechanism | Extend self-evolving to science, law, medicine |
| Pseudo-verification underexplored for domains without ground-truth | TTRL/RLCCF use majority voting for math only | Cross-model agreement + self-consistency for diverse domains |
| Curriculum methods don't generate problems | VCRL/DOTS/AdaCuRL select from fixed pools | Self-generated domain-specific problems at adaptive difficulty |
| RLVR domain extension uses trained verifiers | "Crossing the Reward Bridge" trains a 7B verifier | Training-free pseudo-verification via ensemble consensus |
| No unified framework across domains | Separate solutions per domain | One method instantiable across science, law, medicine |
| Self-play limited to math/games | SQLM, SPIRAL, OpenSIR focus on math/coding/games | Self-play problem generation for arbitrary verifiable domains |

---

## References (Sorted by Relevance)

1. Huang et al. "R-Zero: Self-Evolving Reasoning LLM from Zero Data." arXiv:2508.05004, 2025.
2. Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning." arXiv:2402.03300, 2024.
3. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning via RL." arXiv:2501.12948, 2025.
4. "DAPO: An Open-Source LLM RL System at Scale." arXiv:2503.14476, 2025.
5. "VCRL: Variance-based Curriculum RL for LLMs." arXiv:2509.19803, 2025.
6. "AdaCuRL: Adaptive Curriculum RL with Invalid Sample Mitigation." arXiv:2511.09478, 2025.
7. "DOTS: Difficulty-targeted Online Data Selection." arXiv:2506.05316, 2025.
8. "TTRL: Test-Time Reinforcement Learning." arXiv:2504.16084, 2025.
9. "RLCCF: Wisdom of the Crowd." arXiv:2508.12338, 2025.
10. Yuan et al. "Self-Rewarding Language Models." arXiv:2401.10020, 2024.
11. "PRISM: Post-Training Without Verifiable Rewards." arXiv:2601.04700, 2025.
12. "Crossing the Reward Bridge: Expanding RLVR." arXiv:2503.23829, 2025.
13. Qi et al. "WebRL: Self-Evolving Online Curriculum RL." ICLR 2025.
14. "Self-Questioning Language Models." arXiv:2508.03682, 2025.
15. "OpenSIR: Open-Ended Self-Improving Reasoner." arXiv:2511.00602, 2025.
16. "SPIRAL: Self-Play on Zero-Sum Games." arXiv:2506.24119, 2025.
17. "Multi-Agent Evolve: LLM Self-Improve." arXiv:2510.23595, 2025.
18. "RLVE: Adaptive Verifiable Environments." arXiv:2511.07317, 2025.
19. "WildSci: Scientific Reasoning from Literature." arXiv:2601.05567, 2025.
20. "Med-RLVR." arXiv:2502.19655, 2025.
21. "Genome-Bench." arXiv:2505.19501, 2025.
22. "E2H Reasoner: Curriculum RL Easy to Hard." arXiv:2506.06632, 2025.
23. "MRACL: Multi-Reward Adaptive Curriculum RL." AAAI 2026.
24. "Learning Like Humans: ADCL + EGSR." EMNLP 2025, arXiv:2505.08364.
25. "PRIME: Process Reinforcement through Implicit Rewards." arXiv:2502.01456, 2025.
26. "STILL-3: Online RL Iteration." arXiv:2503.04548, 2025.
27. "SciBench." ICML 2024, arXiv:2307.10635.
28. "MMLU-Pro." NeurIPS 2024, arXiv:2406.01574.
29. "LegalBench." NeurIPS 2024.
30. "Med-PRM: Stepwise Guideline-Verified Process Rewards." arXiv:2506.11474, 2025.
31. "GraphDancer: Training LLMs via Curriculum RL." arXiv:2602.02518, 2026.
32. "Meta-Rewarding Language Models." arXiv:2407.19594, 2024.
