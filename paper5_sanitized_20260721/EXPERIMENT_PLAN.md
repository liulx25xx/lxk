# Experiment Plan: SelfCurriculum

## 1. Research Questions

- **RQ1**: Can self-evolving curriculum learning improve reasoning in non-math domains (science, law, medicine)?
- **RQ2**: Does the Composite Pseudo-Verifier (CPV) provide reliable reward signals across domains?
- **RQ3**: How does adaptive difficulty curriculum compare to fixed-data baselines?
- **RQ4**: What is the contribution of each component (self-generation, CPV, curriculum)?
- **RQ5**: How does performance scale with iteration count and model size?

---

## 2. Evaluation Benchmarks

### 2.1 Primary Benchmarks (6 domains)

| Domain | Benchmark | Size | Format | Metric | Source |
|--------|-----------|------|--------|--------|--------|
| Science (Elementary) | ScienceQA | 4,241 test | MCQ | Accuracy | arXiv:2209.09513 |
| Science (College) | SciBench | 695 | Open-ended | Accuracy | ICML 2024 |
| Science (Reasoning) | ARC-Challenge | 1,172 test | MCQ | Accuracy | AI2 |
| Law | LegalBench | 1,800+ (6 task types) | MCQ/TF | Accuracy per task type | NeurIPS 2024 |
| Medicine | MedQA (USMLE) | 1,273 test | MCQ (5-choice) | Accuracy | arXiv:2009.13081 |
| General Knowledge | MMLU-Pro (subsets) | ~2,000 (selected domains) | MCQ (10-choice) | Accuracy | NeurIPS 2024 |

### 2.2 Math Benchmarks (for comparison with R-Zero)

| Benchmark | Size | Format | Metric |
|-----------|------|--------|--------|
| MATH | 5,000 test | Open-ended | Accuracy |
| GSM8K | 1,319 test | Open-ended | Accuracy |
| AIME 2024 | 30 | Open-ended | Pass@1 |

### 2.3 Transfer/Generalization Benchmarks

| Benchmark | Purpose | Size | Metric |
|-----------|---------|------|--------|
| GPQA (Diamond) | Graduate-level science | 198 | Accuracy |
| PubMedQA | Biomedical QA | 500 test | Accuracy |
| TheoremQA | Theorem application | 800 | Accuracy |

---

## 3. Baselines

### 3.1 No-Training Baselines
1. **Base Model (Zero-Shot)**: Qwen2.5-7B-Instruct / Qwen2.5-14B-Instruct without any additional training. Establishes the floor.
2. **Base Model + CoT**: Same model with chain-of-thought prompting.

### 3.2 SFT Baselines
3. **Domain-SFT**: Supervised fine-tuning on domain-specific training data (same seed data used for our few-shot exemplars, expanded to ~5K per domain via synthetic generation + filtering).
4. **Multi-Domain SFT**: Joint SFT across all domains.

### 3.3 RL Baselines
5. **GRPO (Fixed Data)**: Standard GRPO training on fixed domain datasets with ground-truth rewards (oracle comparison — represents upper bound of what RLVR can achieve with perfect verification).
6. **GRPO + Majority Vote**: GRPO with majority-vote pseudo-labels (ablates CPV by removing cross-model agreement).
7. **R-Zero (Math-Only)**: Reproduce R-Zero on math domain; then evaluate cross-domain transfer.
8. **R-Zero (Domain-Adapted)**: Naively adapt R-Zero to each domain by changing the Challenger's domain prompt (no CPV, no curriculum controller).

### 3.4 DPO Baselines
9. **Self-Rewarding DPO**: Apply Yuan et al.'s self-rewarding iterative DPO to each domain.
10. **DPO + Synthetic Preferences**: Generate preference pairs using CPV scores; train with DPO.

### 3.5 Existing Domain Methods
11. **Med-RLVR** (for medicine): Reproduce their 3B model on MedQA.
12. **WildSci-RL** (for science): Reproduce their RL setup on ScienceQA.

---

## 4. Model Configurations

### 4.1 Primary Experiments (Qwen2.5-7B)

| Configuration | Training Model | Reference Model (CMA) | Domains |
|---------------|---------------|----------------------|---------|
| SelfCurriculum-7B-Science | Qwen2.5-7B | Llama-3.1-8B | Science |
| SelfCurriculum-7B-Law | Qwen2.5-7B | Llama-3.1-8B | Law |
| SelfCurriculum-7B-Med | Qwen2.5-7B | Llama-3.1-8B | Medicine |
| SelfCurriculum-7B-Multi | Qwen2.5-7B | Llama-3.1-8B | All domains jointly |
| SelfCurriculum-7B-Math | Qwen2.5-7B | Llama-3.1-8B | Math (comparison) |

### 4.2 Scaling Experiments

| Configuration | Model Size | GPUs Needed | Purpose |
|---------------|-----------|-------------|---------|
| SelfCurriculum-1.5B | Qwen2.5-1.5B | 8 | Scaling lower bound |
| SelfCurriculum-7B | Qwen2.5-7B | 16 | Main experiments |
| SelfCurriculum-14B | Qwen2.5-14B | 24 | Scaling upper bound |

---

## 5. Ablation Studies

### 5.1 CPV Component Ablation

| Ablation | SC | CMA | Fusion | Expected Finding |
|----------|----|----|--------|-----------------|
| Full CPV | Yes | Yes | Yes | Best overall |
| SC Only | Yes | No | N/A | Good for math, weaker for law/med |
| CMA Only | No | Yes | N/A | Stable but less compute-efficient |
| Majority Vote | Yes (simplified) | No | N/A | R-Zero baseline |
| Random Reward | No | No | Random | Lower bound / sanity check |

### 5.2 Curriculum Strategy Ablation

| Strategy | Description | Expected Finding |
|----------|-------------|-----------------|
| Adaptive (ours) | Target 50% accuracy, adjust per domain | Best overall |
| Fixed Easy | Always generate easy problems | Fast initial gains, plateau early |
| Fixed Hard | Always generate hard problems | Slow/unstable |
| Random | Random difficulty | Baseline (no curriculum benefit) |
| VCRL-style | Variance-based from fixed pool | Good but limited by pool diversity |
| Linear Ramp | Linearly increase difficulty over time | Reasonable but not adaptive |

### 5.3 Iteration Count Ablation

| Iterations | Expected Behavior |
|-----------|-------------------|
| 1 | Significant gain on easy domains |
| 2 | Continued improvement |
| 3 | Diminishing returns begin |
| 5 | Near convergence |
| 8 | Possible overfitting / reward hacking |

### 5.4 Reference Model Choice Ablation

| Reference Model | Family | Size | Expected CMA Quality |
|----------------|--------|------|---------------------|
| Llama-3.1-8B-Instruct | Meta | 8B | Good (different family) |
| Gemma-2-9B-Instruct | Google | 9B | Good (different family) |
| Qwen2.5-3B-Instruct | Alibaba | 3B | Weaker (same family, smaller) |
| GPT-4o-mini (API) | OpenAI | Unknown | Best quality, costs $$$ |
| No reference (SC only) | N/A | N/A | Baseline |

### 5.5 Seed Exemplar Size Ablation

| Seed Size | Domains | Expected Finding |
|-----------|---------|-----------------|
| 0 (zero-shot) | All | Challenger generates generic problems |
| 50 | All | Minimal domain signal |
| 200 (default) | All | Good domain coverage |
| 500 | All | Diminishing returns |
| 1000 | All | Marginal improvement |

### 5.6 Domain Mixing Ablation

| Strategy | Description |
|----------|-------------|
| Joint (ours) | Train all domains simultaneously with adaptive mixing |
| Sequential | Train one domain at a time |
| Single-Domain | Separate model per domain |
| Uniform Mix | Equal weight to all domains |

---

## 6. Analysis Experiments

### 6.1 Pseudo-Verifier Reliability Study

**Design**: On 500 examples per domain with known ground truth, measure:
- SC accuracy (% of times majority vote matches ground truth)
- CMA accuracy (% of times cross-model agrees with ground truth)
- CPV accuracy (combined)
- Agreement rates between SC and CMA

**Expected Results Table:**

| Domain | SC Acc | CMA Acc | CPV Acc | SC-CMA Agreement |
|--------|--------|---------|---------|-------------------|
| Math | ~85% | ~80% | ~90% | ~75% |
| Science (MCQ) | ~75% | ~70% | ~82% | ~65% |
| Law | ~60% | ~65% | ~72% | ~55% |
| Medicine | ~70% | ~72% | ~80% | ~62% |

### 6.2 Generated Problem Quality Analysis

**Design**: Sample 100 generated problems per domain per iteration. Human annotators rate:
1. **Validity**: Is this a well-formed, answerable problem? (Binary)
2. **Domain Faithfulness**: Does it test domain-specific reasoning? (1-5 scale)
3. **Difficulty Calibration**: Is the difficulty appropriate for the target level? (1-5)
4. **Answer Correctness**: Is the provided "correct" answer actually correct? (Binary)

### 6.3 Difficulty Progression Visualization

For each domain, plot:
- Target difficulty $\delta_d^{(t)}$ vs. iteration $t$
- Solver accuracy on current-iteration problems vs. $t$
- Distribution of problem difficulty scores per iteration (histogram)

### 6.4 Reasoning Quality Analysis

**Design**: For 50 test examples per domain, compare reasoning chains from:
- Base model
- SFT baseline
- SelfCurriculum (iteration 1 vs. 5)

Evaluate:
- Reasoning step count
- Logical coherence (human rating 1-5)
- Domain-specific terminology usage
- Error type distribution (factual, logical, calculation)

### 6.5 Cross-Domain Transfer Analysis

Train on single domains, evaluate on others:

| Train → Eval | Science | Law | Medicine | Math | General |
|-------------|---------|-----|----------|------|---------|
| Science | ✓ (in-domain) | ? | ? | ? | ? |
| Law | ? | ✓ | ? | ? | ? |
| Medicine | ? | ? | ✓ | ? | ? |
| Multi-domain | ? | ? | ? | ? | ? |

### 6.6 Failure Mode Analysis

Categorize failures into:
1. **Pseudo-verifier failure**: CPV assigns wrong reward (false positive/negative)
2. **Challenger failure**: Generated problem is invalid or trivial
3. **Solver failure**: Correct approach but execution error
4. **Domain gap**: Problem requires knowledge outside training distribution

---

## 7. Compute Budget

### 7.1 Hardware: 24x H200 (80GB each)

| Experiment | GPUs | Time (hours) | Total GPU-hours |
|------------|------|-------------|-----------------|
| **Main experiments (5 configs × 5 iter)** | 16+4+2+2 | 5 × 35 = 175 | 4,200 |
| **Baselines (12 baselines)** | 16 | 12 × 20 = 240 | 3,840 |
| **Ablations (CPV: 5)** | 16 | 5 × 35 = 175 | 2,800 |
| **Ablations (Curriculum: 6)** | 16 | 6 × 35 = 210 | 3,360 |
| **Ablations (Iterations: 4)** | 16 | 4 × 35 = 140 | 2,240 |
| **Ablations (Ref model: 4)** | 16 | 4 × 35 = 140 | 2,240 |
| **Ablations (Seed size: 4)** | 16 | 4 × 35 = 140 | 2,240 |
| **Scaling (1.5B, 14B)** | 8/24 | 2 × 35 = 70 | 1,120 |
| **Evaluation & analysis** | 4 | 50 | 200 |
| **Total** | — | — | **~22,240** |

### 7.2 Time Budget (5 days)

| Day | Activity | Experiments |
|-----|----------|-------------|
| **Day 1** | Setup + Main experiments | Infrastructure setup, start main 5 configs |
| **Day 2** | Baselines + Main continued | Run all baselines in parallel; main experiments complete |
| **Day 3** | Ablations | CPV ablation, curriculum ablation (parallel groups) |
| **Day 4** | More ablations + Analysis | Iteration/ref model/seed ablations; start analysis |
| **Day 5** | Scaling + Final eval | Scaling experiments; compile all results; figures |

**Parallelism strategy**: With 24 GPUs, can run 2-3 experiments simultaneously (each using 8-16 GPUs).

### 7.3 API Budget ($1K)

| Use | Calls | Cost |
|-----|-------|------|
| GPT-4o-mini CMA (ablation) | 40K | ~$3 |
| GPT-4o evaluation (quality analysis) | 5K | ~$15 |
| Human eval interface (if using API for annotation assist) | 2K | ~$8 |
| **Reserve for debugging/reruns** | — | ~$974 |
| **Total** | — | **~$26 used** |

Most of the budget is reserved — our method is designed to be API-light.

---

## 8. Expected Results & Claims

### 8.1 Primary Claims

**Claim 1**: SelfCurriculum improves reasoning accuracy by 3-8% over GRPO baselines across science, law, and medicine domains.

**Claim 2**: The Composite Pseudo-Verifier provides 5-15% more accurate reward signals than majority voting alone for non-math domains.

**Claim 3**: Adaptive curriculum provides consistent 2-4% improvement over random difficulty selection.

**Claim 4**: Multi-domain joint training shows positive transfer between related domains (e.g., science ↔ medicine).

### 8.2 Expected Results (Conservative)

| Method | ScienceQA | ARC-C | LegalBench | MedQA | MATH | Avg |
|--------|-----------|-------|------------|-------|------|-----|
| Base (Qwen2.5-7B) | 78.0 | 72.0 | 55.0 | 52.0 | 58.0 | 63.0 |
| Domain-SFT | 82.0 | 75.0 | 58.0 | 56.0 | 62.0 | 66.6 |
| GRPO (oracle) | 84.0 | 78.0 | 60.0 | 58.0 | 68.0 | 69.6 |
| R-Zero (math→transfer) | 79.0 | 73.0 | 55.0 | 53.0 | 66.0 | 65.2 |
| R-Zero (domain-adapted) | 81.0 | 75.0 | 57.0 | 55.0 | 66.0 | 66.8 |
| **SelfCurriculum (ours)** | **85.0** | **79.0** | **62.0** | **59.0** | **67.0** | **70.4** |

### 8.3 Success Criteria

- **Minimum viable**: Beat GRPO+majority-vote on ≥3 of 5 non-math domains.
- **Strong result**: Beat all baselines on ≥4 domains AND match R-Zero on math.
- **Best case**: Approach oracle GRPO (ground-truth rewards) performance with pseudo-verification.

---

## 9. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CPV unreliable for legal reasoning | Medium | Focus on MCQ-format legal tasks; add human eval analysis |
| Challenger generates low-quality problems | Medium | Quality filtering via CPV; increase seed exemplars |
| No improvement over SFT baselines | Low | Ensure fair SFT baseline (same data budget); focus on domains where SFT has known weaknesses |
| Reward hacking (verbose but wrong answers) | Medium | DAPO-style overlong penalty; format reward |
| Compute budget exceeded | Low | Prioritize main experiments + key ablations; drop scaling if needed |
| Math results below R-Zero | Medium | Not our main claim; focus narrative on non-math gains |
