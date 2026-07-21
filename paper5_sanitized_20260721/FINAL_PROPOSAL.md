# FINAL PROPOSAL: When Does RLVR Beat SFT?

**Title**: "When Does RLVR Beat SFT? A Controlled Multi-Domain Study of Reinforcement Learning vs Supervised Fine-Tuning for LLM Reasoning"

**Target**: EMNLP 2026 (Main Conference, Long Paper, 8 pages + references)
**Date**: 2026-05-16
**Status**: Implementation-ready

---

## 1. Problem Anchor

### 1.1 What Problem?

Since DeepSeek-R1 (Jan 2025), Reinforcement Learning with Verifiable Rewards (RLVR) -- especially GRPO -- has become the default post-training recipe for LLM reasoning. Hundreds of papers now apply RLVR to every domain: math, code, science, medicine, law. Meanwhile, SFT (Supervised Fine-Tuning) remains the workhorse of most practical deployments.

**Yet nobody has systematically answered the most basic practitioner question**: *"Given a domain, a dataset size, and a target distribution -- should I use RLVR or SFT?"*

### 1.2 What Bottleneck?

The bottleneck is the **lack of a controlled, multi-domain comparison** under matched conditions. Existing evidence is:

| Source | Scope | Limitation |
|--------|-------|------------|
| Med-RLVR (2502.19655) | RLVR vs SFT in medicine | Single domain, single data size |
| "The Invisible Leash" (2507.14843) | RLVR theoretical limits | No SFT comparison, no experiments |
| SRL (ICLR 2026) | Hybrid SFT+RL | Proposes a method, doesn't characterize boundary |
| ReLIFT | Dynamic SFT/RL switching | Engineering solution, not understanding |
| Tsinghua Critique (2504.13837) | RLVR limitations | Argues reasoning is locked, no controlled study |
| "Delay, Plateau, Collapse" (2605.02909) | Verifier noise effects | Studies noise, not RLVR vs SFT |

Every comparison is either single-domain, single-scale, or confounded by different base models/data.

### 1.3 Constraints

- **Hardware**: 24x NVIDIA H200 GPUs (80GB each) = 1,920 GB total VRAM
- **API Budget**: $1,000 (for SFT training data generation and evaluation)
- **Timeline**: 5 days from start to completion
- **Base Model**: Qwen2.5-7B-Instruct (primary), optionally 3B/14B for scaling
- **Page limit**: 8 pages + references

### 1.4 Success Condition

The paper succeeds if it delivers:
1. A clear empirical answer to "when does RLVR beat SFT?" across multiple axes (domain, difficulty, data size, distribution shift)
2. Non-trivial crossover patterns (not "one always wins")
3. An actionable decision framework that practitioners can use immediately
4. Controlled experimental design that reviewers cannot dismiss as confounded

**Minimum viable result**: RLVR wins on some conditions, SFT wins on others, with identifiable patterns.

**Ideal result**: Clean "RLVR Benefit Frontier" showing crossover as a function of difficulty x data size x domain, with the hybrid SFT->RLVR beating both.

---

## 2. Core Thesis and Hypotheses

### 2.1 Core Thesis

RLVR and SFT are complementary, not competing, post-training paradigms. Their relative advantage depends systematically on **four factors**: (1) whether the task requires novel knowledge vs. reasoning reorganization, (2) task difficulty relative to the base model's capability, (3) training data availability, and (4) whether the target is in-distribution or out-of-distribution generalization.

### 2.2 Testable Hypotheses

| ID | Hypothesis | Operationalization | Expected Finding |
|----|-----------|-------------------|------------------|
| **H1** | **Domain Effect**: RLVR advantage varies by knowledge density | Compare Delta_RLVR = Acc(GRPO) - Acc(SFT) across 6 domains | RLVR wins in procedural domains (math, code); SFT wins in knowledge-intensive domains (medicine, law) |
| **H2** | **Difficulty Effect**: RLVR peaks at moderate difficulty | Stratify each domain by base-model accuracy (easy/medium/hard) | RLVR advantage largest at medium difficulty (30-60% base accuracy); SFT wins at hard tasks (<30%) where RLVR gets no positive signal |
| **H3** | **Data Efficiency**: RLVR is more sample-efficient | Vary training data from 100 to 10K | RLVR > SFT at N<500; SFT catches up at N>2K; crossover depends on domain |
| **H4** | **OOD Generalization**: RLVR generalizes better | Evaluate on held-out OOD benchmarks per domain | RLVR consistently better OOD, even in domains where SFT wins in-distribution |
| **H5** | **Hybrid Superiority**: SFT->RLVR beats both | Run sequential SFT then GRPO | SFT provides knowledge, RLVR refines reasoning; hybrid is Pareto-optimal |
| **H6** | **Compute-Performance Tradeoff**: SFT is more compute-efficient | Match total FLOPs, compare performance | SFT converges faster, RLVR produces more robust models |

---

## 3. Experimental Design

### 3.1 Domains and Benchmarks

We select 6 domains spanning the reasoning spectrum from purely procedural (math) to heavily knowledge-dependent (medicine, law):

| Domain | Reasoning Type | Knowledge Dependence | Training Source | N_train | In-Domain Test | N_test | OOD Test |
|--------|---------------|---------------------|----------------|---------|---------------|--------|----------|
| **Math** | Procedural/symbolic | Low | MATH train | 7,500 | MATH test | 5,000 | GSM8K (1,319), AMC 2023 |
| **Code** | Procedural/logical | Low-Medium | MBPP train + APPS intro | ~2,500 | MBPP test (500), HumanEval (164) | 664 | LiveCodeBench (recent), CodeContests-easy |
| **Science** | Analytical/factual | Medium | ARC-C train + ScienceQA (text-only) | ~5,000 | ARC-C test (1,172), ScienceQA test | ~3,000 | GPQA Diamond (198), SciBench (695) |
| **Medicine** | Clinical reasoning | High | MedQA train (USMLE) | 10,178 | MedQA test | 1,273 | PubMedQA (500), MMLU-Medical (~1,000) |
| **Law** | Rule application | High | LegalBench (6 task types) | ~3,000 | LegalBench test | ~1,800 | Bar exam MCQ subset, MMLU-Law |
| **Commonsense** | Intuitive inference | Medium | HellaSwag train (subsample) + ARC-E | ~8,000 | HellaSwag val (10K), ARC-E test (2,376) | ~12,000 | WinoGrande (1,267), PIQA (1,838) |

**Domain Selection Rationale**:
- **Math & Code**: RLVR's "home turf" -- natural verifiers exist (correctness check, test execution). Establishes the upper bound of RLVR benefit.
- **Science & Commonsense**: Mixed -- verifiable answer format (MCQ) but moderate knowledge demand.
- **Medicine & Law**: RLVR's "away game" -- requires substantial domain knowledge that may not be in the base model.

### 3.2 Training Methods

We compare **three training paradigms** plus **two hybrid conditions**:

#### A. RLVR (GRPO) -- Reinforcement Learning with Verifiable Rewards

The model generates responses and receives binary reward based on answer correctness:

```
Input: question Q (no answer provided to model)
Process: Generate G responses {y_1, ..., y_G} from pi_theta(.|Q)
Reward: r_i = 1 if extract_answer(y_i) == ground_truth else 0
Update: GRPO policy gradient with group-normalized advantages
```

**Reward functions by domain**:
- Math: Exact match of numerical/symbolic answer (after normalization)
- Code: Binary -- all test cases pass (1) or any fail (0)
- MCQ domains (Science, Medicine, Law, Commonsense): Exact match of answer letter (A/B/C/D/E)

#### B. SFT -- Supervised Fine-Tuning

Standard next-token-prediction training on (question, chain-of-thought + answer) pairs:

```
Input: (Q, CoT_reasoning + correct_answer) pair
Process: Minimize cross-entropy loss on the response tokens
```

**CoT data construction** (critical for fairness -- SFT gets the best possible demonstrations):
- Math: Use existing step-by-step solutions from MATH dataset
- Code: Use canonical solutions from MBPP/APPS
- MCQ domains: Generate CoT via rejection sampling from Qwen2.5-72B-Instruct:
  1. For each (question, correct_answer) pair, generate 8 CoT responses
  2. Keep only responses that arrive at the correct answer
  3. Select the most concise correct response
  4. Expected yield: ~85-95% of examples will have at least one correct CoT
  5. For remaining ~5-15%, use the correct answer without CoT

#### C. DPO -- Direct Preference Optimization

Preference learning from self-generated rollouts:

```
Input: question Q
Process: Generate 8 responses from base model
          Correct answer -> preferred; Wrong answer -> rejected
          Form preference pairs (y_w, y_l) per question
Update: DPO loss with beta=0.1
```

#### D. Hybrid: SFT -> GRPO (Sequential)

1. First: SFT for E_sft epochs (until convergence on training data)
2. Then: GRPO for E_rl steps (using same training questions as prompts)

#### E. Hybrid: GRPO -> SFT (Sequential)

1. First: GRPO for E_rl steps
2. Then: SFT for E_sft epochs on the same training data

### 3.3 Fairness Protocol (Critical for Credibility)

This is the most important design decision. We ensure apples-to-apples comparison along three axes:

#### Same Data
- All methods train on the **exact same N training questions** per condition
- SFT additionally has access to correct answers + CoT demonstrations
- GRPO has access to correct answers only as reward signal (not shown to model)
- DPO has access to correct answers for labeling preferences
- This asymmetry is **intentional** -- it reflects the real-world tradeoff: SFT has richer supervision, RLVR has weaker but more scalable supervision

#### Same Base Model
- All methods start from the same checkpoint: **Qwen2.5-7B-Instruct**
- No intermediate pretraining or additional data mixing
- All models use the same tokenizer, same chat template

#### Same Evaluation
- All methods evaluated with:
  - Temperature 0.0 (greedy decoding)
  - Same prompt template per domain
  - Pass@1 accuracy (single attempt)
  - CoT reasoning enabled for all (no unfair prompting advantage)

#### Compute Reporting
- We report: training time (GPU-hours), total FLOPs, number of gradient steps, total tokens processed
- **Primary comparison**: same data budget, each trained to convergence
- **Secondary analysis**: performance per unit compute (efficiency curves)

### 3.4 Experimental Conditions Matrix

#### Tier 1: Core Comparison (MUST DO -- 18 runs)

All 6 domains x 3 methods (SFT, GRPO, DPO) at moderate data size.

| Domain | N_train | SFT | GRPO | DPO |
|--------|---------|-----|------|-----|
| Math | 5,000 | Run-01 | Run-02 | Run-03 |
| Code | 2,500 | Run-04 | Run-05 | Run-06 |
| Science | 5,000 | Run-07 | Run-08 | Run-09 |
| Medicine | 5,000 | Run-10 | Run-11 | Run-12 |
| Law | 3,000 | Run-13 | Run-14 | Run-15 |
| Commonsense | 5,000 | Run-16 | Run-17 | Run-18 |

#### Tier 2: Data-Size Ablation (HIGH PRIORITY -- 18 new runs)

3 representative domains x 3 additional data sizes x 2 methods (SFT, GRPO):

| Data Size | Math-SFT | Math-GRPO | Medicine-SFT | Medicine-GRPO | Science-SFT | Science-GRPO |
|-----------|----------|-----------|-------------|--------------|-------------|--------------|
| 100 | Run-19 | Run-20 | Run-25 | Run-26 | Run-31 | Run-32 |
| 500 | Run-21 | Run-22 | Run-27 | Run-28 | Run-33 | Run-34 |
| 2,000 | Run-23 | Run-24 | Run-29 | Run-30 | Run-35 | Run-36 |
| 5,000 | (=Tier1) | (=Tier1) | (=Tier1) | (=Tier1) | (=Tier1) | (=Tier1) |

#### Tier 3: Hybrid Methods (PRIORITY -- 6 runs)

3 domains x 2 hybrid strategies:

| Domain | SFT->GRPO | GRPO->SFT |
|--------|-----------|-----------|
| Math | Run-37 | Run-38 |
| Medicine | Run-39 | Run-40 |
| Science | Run-41 | Run-42 |

#### Tier 4: Scaling (OPTIONAL -- 8 runs)

2 domains x 2 sizes x 2 methods:

| Model Size | Math-SFT | Math-GRPO | Medicine-SFT | Medicine-GRPO |
|-----------|----------|-----------|-------------|--------------|
| 3B (Qwen2.5-3B) | Run-43 | Run-44 | Run-45 | Run-46 |
| 14B (Qwen2.5-14B) | Run-47 | Run-48 | Run-49 | Run-50 |

#### Total Run Count

| Tier | Runs | Priority |
|------|------|----------|
| Tier 1: Core | 18 | MUST |
| Tier 2: Data-size | 18 | HIGH |
| Tier 3: Hybrid | 6 | HIGH |
| Tier 4: Scaling | 8 | OPTIONAL |
| **Total** | **50** | -- |

### 3.5 Difficulty Stratification (Post-Hoc Analysis, No Extra Training)

For each domain, we stratify test examples by base model (zero-shot) accuracy into three buckets:

| Bucket | Base Model Accuracy | Interpretation |
|--------|-------------------|----------------|
| **Easy** | >70% | Model already knows this -- improvement is refinement |
| **Medium** | 30-70% | "Within reach" -- model sometimes gets it right |
| **Hard** | <30% | "Beyond reach" -- model rarely solves this |

For Math specifically, we use the built-in difficulty levels (Levels 1-5).

This stratification is computed ONCE from the base model's zero-shot performance and applied to all methods' results. No additional training runs needed.

---

## 4. Training Infrastructure

### 4.1 GRPO Configuration

```yaml
# GRPO (Group Relative Policy Optimization) -- following DeepSeek-R1/DAPO
framework: veRL  # or OpenRLHF as fallback

model:
  base: Qwen2.5-7B-Instruct
  max_length: 2048  # response generation
  dtype: bfloat16

grpo:
  group_size: 8  # G = 8 rollouts per prompt
  clip_ratio_low: 0.2  # DAPO asymmetric clipping
  clip_ratio_high: 0.28
  kl_coeff: 0.001  # light KL penalty to base model
  entropy_bonus: 0.001  # prevent entropy collapse
  discount: 1.0  # no discounting for bandit reward

optimizer:
  type: AdamW
  lr: 5e-7
  weight_decay: 0.01
  warmup_ratio: 0.05
  scheduler: cosine
  total_steps: varies_by_data_size  # see table below

rollout:
  temperature: 1.0
  top_p: 1.0
  max_new_tokens: 2048  # math/code
  # max_new_tokens: 1024  # MCQ domains (shorter)

reward:
  type: binary  # r = {0, 1}
  format_bonus: 0.0  # no format reward (keep it clean)

batch:
  prompts_per_batch: 128  # number of unique prompts per update
  gradient_accumulation: 4  # effective batch = 128 * 4 = 512 prompts
  # Total rollouts per update = 512 * 8 = 4,096 sequences

hardware:
  gpus: 8  # per run (3 runs in parallel on 24 GPUs)
  # 4 GPUs for model training (FSDP/DeepSpeed ZeRO-3)
  # 4 GPUs for vLLM rollout generation
```

**Training steps by data size** (targeting ~3 passes through the data):

| N_train | Steps | Effective Prompts | Wall Time (est.) |
|---------|-------|------------------|-----------------|
| 100 | 50 | 25,600 | ~1 hour |
| 500 | 150 | 76,800 | ~2 hours |
| 2,000 | 400 | 204,800 | ~5 hours |
| 5,000 | 800 | 409,600 | ~8 hours |
| 10,000 | 1,200 | 614,400 | ~12 hours |

### 4.2 SFT Configuration

```yaml
# Standard Supervised Fine-Tuning
framework: transformers + DeepSpeed ZeRO-2

model:
  base: Qwen2.5-7B-Instruct
  max_length: 2048
  dtype: bfloat16

training:
  lr: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  scheduler: cosine
  epochs: 3  # train to convergence (early stopping on val loss)
  batch_size: 32
  gradient_accumulation: 2  # effective batch = 64

data:
  format: chatml  # Qwen2.5 chat template
  # Input: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n
  # Target: {chain_of_thought}\n\nThe answer is {answer}.<|im_end|>
  loss_mask: response_only  # only compute loss on assistant tokens

hardware:
  gpus: 4  # per run (6 runs in parallel on 24 GPUs)
```

**Training time estimates**:

| N_train | Epochs | Wall Time (est.) |
|---------|--------|-----------------|
| 100 | 3 | ~15 min |
| 500 | 3 | ~30 min |
| 2,000 | 3 | ~1.5 hours |
| 5,000 | 3 | ~3 hours |
| 10,000 | 3 | ~6 hours |

### 4.3 DPO Configuration

```yaml
# Direct Preference Optimization
framework: TRL (trl library) + DeepSpeed

model:
  base: Qwen2.5-7B-Instruct
  ref_model: Qwen2.5-7B-Instruct  # frozen reference
  dtype: bfloat16

dpo:
  beta: 0.1
  loss_type: sigmoid  # standard DPO

preference_data_construction:
  # For each training question:
  # 1. Generate 8 responses from base model (temperature=1.0)
  # 2. Check answer correctness against ground truth
  # 3. Form pairs: (correct_response, incorrect_response)
  # 4. Require at least 1 correct + 1 incorrect per question
  # 5. Drop questions where all 8 are correct or all 8 are wrong
  rollouts_per_question: 8
  temperature: 1.0

optimizer:
  lr: 5e-7
  warmup_ratio: 0.1
  scheduler: cosine
  epochs: 1

hardware:
  gpus: 4  # per run
```

### 4.4 SFT Training Data Construction Pipeline

```python
# Pseudocode for SFT CoT data generation

def construct_sft_data(questions, answers, domain,
                       generator_model="Qwen2.5-72B-Instruct"):
    """
    Generate Chain-of-Thought training data for SFT.
    Uses rejection sampling from a strong model.
    """
    sft_data = []
    
    for q, a in zip(questions, answers):
        # Generate K candidate CoT responses
        K = 8
        candidates = generator_model.generate(
            prompt=format_cot_prompt(q, domain),
            n=K,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Filter for correct final answers
        correct_cots = [c for c in candidates
                        if extract_answer(c) == a]
        
        if correct_cots:
            # Select the most concise correct CoT
            best_cot = min(correct_cots, key=len)
            sft_data.append({
                "question": q,
                "response": best_cot
            })
        else:
            # Fallback: use direct answer without CoT
            sft_data.append({
                "question": q,
                "response": f"The answer is {a}."
            })
    
    return sft_data

# Domain-specific prompt templates for CoT generation:
PROMPTS = {
    "math": (
        "Solve the following math problem step by step. "
        "Show your reasoning, then give the final numerical answer."
        "\n\nProblem: {question}\n\nSolution:"
    ),
    "code": (
        "Write a Python solution for the following problem. "
        "Think through your approach step by step, then write the code."
        "\n\nProblem: {question}\n\nSolution:"
    ),
    "science": (
        "Answer the following science question. Explain your reasoning "
        "step by step, then select the correct answer."
        "\n\nQuestion: {question}\n\nReasoning:"
    ),
    "medicine": (
        "Answer the following medical question. Think through the "
        "clinical reasoning step by step, then select the best answer."
        "\n\nQuestion: {question}\n\nClinical reasoning:"
    ),
    "law": (
        "Answer the following legal reasoning question. Analyze the "
        "legal principles involved step by step, then select the correct answer."
        "\n\nQuestion: {question}\n\nLegal analysis:"
    ),
    "commonsense": (
        "Answer the following question using common sense reasoning. "
        "Explain your reasoning, then select the correct answer."
        "\n\nQuestion: {question}\n\nReasoning:"
    ),
}
```

**SFT Data Construction Cost**:
- Total questions needing CoT generation: ~15,000 (MCQ domains only; math/code have existing solutions)
- Using Qwen2.5-72B-Instruct locally (4 GPUs with AWQ quantization): FREE (local inference)
- Backup: GPT-4o-mini API, 15K x 8 samples = 120K calls, ~$25
- Time: ~4-6 hours on 4 GPUs for local generation

### 4.5 Hardware Allocation Plan

```
24x H200 GPU Allocation:

Option A: Maximum Parallelism (SFT phase)
  GPUs  0-3:  SFT Run 1
  GPUs  4-7:  SFT Run 2
  GPUs  8-11: SFT Run 3
  GPUs 12-15: SFT Run 4
  GPUs 16-19: SFT Run 5
  GPUs 20-23: SFT Run 6
  => 6 SFT runs in parallel

Option B: GRPO Phase
  GPUs  0-7:  GRPO Run 1 (4 train + 4 vLLM)
  GPUs  8-15: GRPO Run 2 (4 train + 4 vLLM)
  GPUs 16-23: GRPO Run 3 (4 train + 4 vLLM)
  => 3 GRPO runs in parallel

Option C: Mixed (SFT + GRPO)
  GPUs  0-7:  GRPO Run (8 GPUs)
  GPUs  8-15: GRPO Run (8 GPUs)
  GPUs 16-19: SFT Run 1 (4 GPUs)
  GPUs 20-23: SFT Run 2 (4 GPUs)
  => 2 GRPO + 2 SFT in parallel
```

---

## 5. Evaluation Protocol

### 5.1 In-Domain Evaluation

| Domain | Benchmark | N | Format | Metric |
|--------|-----------|---|--------|--------|
| Math | MATH test | 5,000 | Open-ended | Accuracy (exact match) |
| Code | MBPP test | 500 | Code generation | Pass@1 (execution) |
| Code | HumanEval | 164 | Code generation | Pass@1 (execution) |
| Science | ARC-Challenge test | 1,172 | 4-choice MCQ | Accuracy |
| Science | ScienceQA test (text) | ~2,000 | MCQ | Accuracy |
| Medicine | MedQA test | 1,273 | 5-choice MCQ | Accuracy |
| Law | LegalBench test | ~1,800 | MCQ/TF | Accuracy (per task + avg) |
| Commonsense | HellaSwag val | 10,042 | 4-choice MCQ | Accuracy |
| Commonsense | ARC-Easy test | 2,376 | 4-choice MCQ | Accuracy |

### 5.2 Out-of-Distribution (OOD) Evaluation

| Domain | OOD Benchmark | N | Why OOD |
|--------|--------------|---|---------|
| Math | GSM8K test | 1,319 | Different difficulty distribution (easier, word problems) |
| Math | AMC 2023 / AIME 2024 | 30-40 | Much harder, competition math |
| Code | LiveCodeBench (recent) | ~200 | Unseen problems, post-training-cutoff |
| Science | GPQA Diamond | 198 | Graduate-level, much harder |
| Science | SciBench | 695 | College-level, open-ended |
| Medicine | PubMedQA | 500 | Different format (yes/no/maybe from abstracts) |
| Medicine | MMLU-Medical | ~1,000 | Different exam style |
| Law | MMLU-Law | ~400 | Different legal reasoning style |
| Commonsense | WinoGrande | 1,267 | Pronoun resolution (different task type) |
| Commonsense | PIQA | 1,838 | Physical intuition (different reasoning type) |

### 5.3 Evaluation Details

```yaml
evaluation:
  decoding:
    temperature: 0.0  # greedy for all
    max_new_tokens: 2048  # math/code; 1024 for MCQ
    
  answer_extraction:
    math: regex for boxed answer \boxed{...} or last number
    code: execute against test cases
    mcq: regex for letter (A|B|C|D|E) in last sentence
    
  prompt_template:
    # Same for all methods -- no method-specific prompting
    math: "Solve the following math problem step by step.
           \n\n{question}\n\nSolution:"
    code: "Write a Python function to solve the problem.
           \n\n{question}\n\nSolution:"
    mcq:  "{question}\n\nLet's think step by step."
    
  seeds: 1  # greedy (deterministic), no need for multiple seeds
  # For statistical significance: bootstrap 95% CI on test set accuracy
```

---

## 6. Analysis Plan

### 6.1 Main Comparison (-> Table 1 + Figure 1)

**Table 1**: The "money table" -- 6 domains x 5 methods (Base, SFT, GRPO, DPO, SFT->GRPO).

```
Table 1: Main Results (Accuracy %) -- Qwen2.5-7B-Instruct, ~5K training examples per domain

                    | Math  | Code  | Science | Medicine | Law   | CS    | Avg
--------------------|-------|-------|---------|----------|-------|-------|-----
Base (zero-shot)    |  58.0 | 48.0  |  72.0   |  52.0    | 55.0  | 75.0  | 60.0
SFT                 |  65.0 | 60.0  |  78.0   |  60.0    | 62.0  | 80.0  | 67.5
DPO                 |  62.0 | 55.0  |  76.0   |  56.0    | 58.0  | 78.0  | 64.2
GRPO                |  70.0 | 65.0  |  77.0   |  57.0    | 58.0  | 79.0  | 67.7
SFT->GRPO           |  72.0 | 67.0  |  80.0   |  61.0    | 63.0  | 81.0  | 70.7
```

*(Numbers are illustrative estimates based on prior work; actual numbers from experiments.)*

**Figure 1**: Bar chart showing Delta_RLVR = Acc(GRPO) - Acc(SFT) per domain. Color-coded: green = RLVR wins, red = SFT wins.

**Expected pattern**: Delta_RLVR > 0 for math and code (procedural), Delta_RLVR ~ 0 for science and commonsense, Delta_RLVR < 0 for medicine and law (knowledge-intensive).

### 6.2 Data Efficiency Analysis (-> Figure 2)

**Figure 2**: Three panels (Math, Medicine, Science). Each panel: X-axis = training data size (100, 500, 2K, 5K); Y-axis = test accuracy. Two lines: SFT (blue) and GRPO (red). Shaded area between = RLVR advantage/disadvantage.

```
Expected pattern:

Math:
  N=100:  GRPO >> SFT  (GRPO: 62%, SFT: 59%)
  N=500:  GRPO > SFT   (GRPO: 66%, SFT: 62%)
  N=2K:   GRPO > SFT   (GRPO: 69%, SFT: 64%)
  N=5K:   GRPO >= SFT  (GRPO: 70%, SFT: 65%)
  
Medicine:
  N=100:  GRPO ~ SFT   (GRPO: 53%, SFT: 53%)
  N=500:  SFT > GRPO   (SFT: 55%, GRPO: 54%)
  N=2K:   SFT > GRPO   (SFT: 58%, GRPO: 56%)
  N=5K:   SFT >> GRPO  (SFT: 60%, GRPO: 57%)
  
Science:
  N=100:  GRPO > SFT   (crossover at ~1K)
  N=5K:   SFT ~ GRPO   (roughly equal)
```

**Key finding**: The "crossover point" (data size where SFT catches up to GRPO) varies by domain -- earlier for knowledge-intensive domains, later for procedural domains.

### 6.3 OOD Generalization Analysis (-> Figure 3 + Table 2)

**Table 2**: OOD performance comparison.

```
Table 2: In-Distribution vs Out-of-Distribution Performance (Accuracy %)

                   | In-Domain          | Out-of-Domain          | OOD Gap (lower=better)
                   | SFT   | GRPO      | SFT   | GRPO           | SFT    | GRPO
Math               | 65.0  | 70.0      | 72.0* | 76.0* (GSM8K)  | -7.0   | -6.0
Medicine           | 60.0  | 57.0      | 42.0  | 48.0 (PubMedQA)| 18.0   |  9.0
Science            | 78.0  | 77.0      | 35.0  | 40.0 (GPQA)    | 43.0   | 37.0
```

*GSM8K is "easier" OOD, so scores go up.

**Figure 3**: Grouped bar chart. For each domain: In-domain SFT, In-domain GRPO, OOD SFT, OOD GRPO. Shows GRPO's OOD advantage.

**Expected finding**: Even in domains where SFT wins in-distribution (medicine, law), GRPO has a smaller OOD gap -- suggesting RLVR learns more generalizable reasoning strategies.

### 6.4 Difficulty Stratification Analysis (-> Figure 4)

**Figure 4**: For each domain, a 3-group bar chart (Easy/Medium/Hard x SFT vs GRPO).

```
Expected pattern for Math (MATH levels):

Level 1-2 (Easy):    SFT ~ GRPO ~ 90%   (both saturate)
Level 3 (Medium):    GRPO > SFT by ~8%   (RLVR peak advantage)
Level 4 (Hard):      GRPO > SFT by ~5%   (advantage narrows)
Level 5 (Very Hard): GRPO ~ SFT ~ 20%    (both fail, near base model)
```

**Key insight**: RLVR's maximum advantage occurs at the "within-reach" difficulty level, where the base model occasionally succeeds and can be reinforced.

### 6.5 The RLVR Benefit Frontier (-> Figure 5 -- THE SIGNATURE FIGURE)

**Figure 5**: 2D heatmap. X-axis = data size (log scale: 100, 500, 2K, 5K). Y-axis = task difficulty (Easy, Medium, Hard). Color = Delta_RLVR = Acc(GRPO) - Acc(SFT). One panel per domain.

The **zero contour line** is the "RLVR Benefit Frontier" -- the boundary in (data_size, difficulty) space where RLVR transitions from beneficial to detrimental.

**Expected shape**: The frontier slopes downward-right -- RLVR is beneficial for harder tasks with less data, and SFT takes over for easier tasks with abundant data.

```
                    +------------------------------+
  Hard              |  RLVR ~ SFT    |  SFT wins   |
  (base acc <30%)   |  (both fail)   |  (SFT has   |
                    |                |  edge with  |
                    +----------------|  more data) |
  Medium            |  RLVR wins     |             |
  (base acc 30-70%) |  (peak benefit)|   --------  |
                    |                |  Frontier   |
                    +----------------+             |
  Easy              |  RLVR ~ SFT   |  SFT wins   |
  (base acc >70%)   |  (both good)   |  slightly   |
                    +------------------------------+
                     100   500   2K    5K    10K
                            Training Data Size
```

### 6.6 Hybrid Analysis (-> Table 3)

**Table 3**: Hybrid methods on 3 representative domains.

```
Table 3: Hybrid SFT + GRPO Results (Accuracy %)

                   | Math  | Medicine | Science |
SFT only           | 65.0  | 60.0     | 78.0    |
GRPO only          | 70.0  | 57.0     | 77.0    |
DPO only           | 62.0  | 56.0     | 76.0    |
SFT -> GRPO        | 72.0  | 61.0     | 80.0    |
GRPO -> SFT        | 68.0  | 59.0     | 79.0    |
```

**Expected finding**: SFT->GRPO consistently beats both individual methods. The order matters: SFT first provides knowledge scaffolding, then RLVR refines reasoning. GRPO->SFT is weaker (SFT may overwrite learned reasoning patterns).

### 6.7 Compute Efficiency Analysis (-> Figure 6)

**Figure 6**: X-axis = GPU-hours. Y-axis = accuracy. Lines for SFT (converges fast, plateaus early) and GRPO (slower start, higher ceiling or similar ceiling depending on domain).

Report total FLOPs for each method per domain:

```
Approximate FLOPs per training run (Qwen2.5-7B, N=5K):

SFT:  3 epochs * 5K examples * ~2K tokens/example * 7B params * 6 FLOPs/token
    ~ 1.26 * 10^18 FLOPs ~ 1.26 ExaFLOPs

GRPO: 800 steps * 512 prompts * 8 rollouts * ~1K tokens * 7B * 6
    ~ 137 * 10^18 FLOPs ~ 137 ExaFLOPs
    (dominated by rollout generation)

=> GRPO uses ~100x more FLOPs than SFT
=> If GRPO wins by only 2-3% on procedural domains, is it worth the compute?
```

This analysis adds nuance: RLVR may be "better" but at much higher cost. When is the extra compute justified?

### 6.8 Additional Analyses

**A. Reasoning Chain Analysis**: For 100 test examples per domain, compare:
- Average CoT length (tokens)
- Reasoning step count
- Error types: factual error, logical error, calculation error, format error
- Self-consistency (pass@1 vs pass@8 gap)

**B. Training Dynamics**: Plot accuracy-on-test-set vs training steps for both SFT and GRPO per domain. Expected: SFT jumps quickly then plateaus; GRPO rises slowly but may reach higher.

**C. Reward Signal Quality**: For GRPO runs, report the reward distribution statistics per domain -- what fraction of rollouts get reward=1? If very low (<10%), RLVR struggles because there's insufficient positive signal.

---

## 7. Expected Findings and Story Arc

### 7.1 Primary Findings (Expected)

**Finding 1: Domain Dichotomy**
> RLVR (GRPO) outperforms SFT on procedural reasoning domains (math: +5%, code: +5%), roughly matches on analytical domains (science, commonsense), and underperforms on knowledge-intensive domains (medicine: -3%, law: -4%).

**Finding 2: The Data-Size Crossover**
> In data-scarce regimes (N < 500), RLVR is competitive across all domains because it makes better use of limited signal. As data grows, SFT's advantage in knowledge-intensive domains widens, while RLVR maintains its edge in procedural domains.

**Finding 3: OOD Robustness**
> RLVR-trained models consistently show better out-of-distribution generalization across all domains, even those where SFT wins in-distribution. The OOD gap (in-domain minus OOD performance) is 5-15% smaller for RLVR.

**Finding 4: Difficulty Sweet Spot**
> RLVR's maximum advantage occurs at moderate difficulty (base model accuracy 30-60%). On very easy tasks, both methods saturate. On very hard tasks, RLVR fails to learn because it never generates correct responses for positive reinforcement.

**Finding 5: Hybrid Is Best**
> Sequential SFT->RLVR outperforms both individual methods across all domains (avg +3-5% over best single method), suggesting they learn complementary aspects: SFT imparts knowledge and format, RLVR refines reasoning strategies.

**Finding 6: Compute-Performance Tradeoff**
> GRPO requires ~100x the FLOPs of SFT. Per compute unit, SFT is more efficient. RLVR is justified only when (a) the task is procedural, (b) OOD generalization matters, or (c) the hybrid pipeline is used.

### 7.2 Practical Decision Framework (Contribution 3)

```
                    PRACTITIONER'S DECISION TREE

                      +--------------------+
                      | You have a domain  |
                      | and labeled data   |
                      +--------+-----------+
                               |
                  +------------v------------+
                  | Is the task procedural? |
                  | (math, code, logic)     |
                  +------+----------+-------+
                    YES  |          |  NO
                         |          |
            +------------v--+  +---v------------------+
            | Use RLVR      |  | Does it require      |
            | (or SFT->RLVR |  | domain knowledge?    |
            |  for best     |  | (medicine, law)      |
            |  results)     |  +---+----------+-------+
            +---------------+   YES|          | NO
                                   |          |
                      +------------v--+  +---v------------------+
                      | How much data? |  | Use RLVR or SFT     |
                      +--+-------+----+  | (roughly same result)|
                    >2K  |       |<500   +----------------------+
                         |       |
            +------------v--+  +-v------------------+
            | Use SFT       |  | Use RLVR or       |
            | (or SFT->RLVR)|  | SFT->RLVR         |
            +---------------+  +--------------------+
                     
  ALWAYS: If OOD generalization matters, add RLVR stage after SFT.
```

### 7.3 Story Arc for the Paper

1. **Hook**: "Everyone uses RLVR now, but should they?"
2. **Setup**: Define the comparison space (domain x data x difficulty x distribution)
3. **Twist**: RLVR doesn't always win -- the winner depends on conditions in predictable ways
4. **Resolution**: Here's exactly when to use what, backed by controlled experiments
5. **Takeaway**: SFT->RLVR is the safest default; pure RLVR only for procedural/low-data settings

---

## 8. Visualization Plan

| Figure | Type | Content | Section | Priority |
|--------|------|---------|---------|----------|
| **Fig 1** | Overview | Experimental design diagram: domains x methods matrix | S1/S3 | MUST |
| **Fig 2** | Line plot | Data efficiency curves: accuracy vs N for SFT/GRPO, 3 domains | S4.2 | MUST |
| **Fig 3** | Grouped bar | In-domain vs OOD: SFT vs GRPO per domain | S4.3 | MUST |
| **Fig 4** | Grouped bar | Difficulty stratification: Easy/Med/Hard x SFT/GRPO | S4.4 | HIGH |
| **Fig 5** | Heatmap | **RLVR Benefit Frontier**: 2D map (data x difficulty -> Delta_RLVR) | S4.5 | MUST (signature) |
| **Fig 6** | Line plot | Compute efficiency: accuracy vs GPU-hours | S4.7 | HIGH |
| **Tab 1** | Table | Main results: 6 domains x 5 methods | S4.1 | MUST |
| **Tab 2** | Table | OOD results | S4.3 | MUST |
| **Tab 3** | Table | Hybrid results | S4.6 | HIGH |
| **Tab 4** | Table | Compute cost comparison | S4.7 | HIGH |

---

## 9. Timeline and Compute Budget

### 9.1 Five-Day Schedule

```
===================================================================
  DAY 0 (Prep, evening before): Data & Infrastructure Setup
===================================================================

Tasks:
  [ ] Download all datasets (MATH, GSM8K, ARC-C, ScienceQA, MedQA,
      LegalBench, MBPP, APPS, HellaSwag, ARC-E, GPQA, SciBench, 
      PubMedQA, WinoGrande, PIQA, LiveCodeBench, MMLU subsets)
  [ ] Subsample training sets at {100, 500, 2K, 5K} for each domain
  [ ] Install veRL / OpenRLHF, verify GRPO training works on toy task
  [ ] Install vLLM, verify inference for Qwen2.5-7B
  [ ] Download Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct (AWQ), 
      Qwen2.5-3B-Instruct, Qwen2.5-14B-Instruct weights
  [ ] Prepare prompt templates for all domains
  [ ] Write evaluation scripts (answer extraction + accuracy)
  [ ] Write data formatting scripts (SFT, DPO, GRPO formats)

GPU Usage: 4 GPUs (for testing)
Time: ~4-6 hours

===================================================================
  DAY 1: SFT Data Generation + All SFT Training + Base Eval
===================================================================

Morning (8:00-12:00):
  [ ] [4 GPUs] Generate CoT data using Qwen2.5-72B-Instruct (AWQ)
      for MCQ domains via rejection sampling (~15K questions * 8 samples)
  [ ] [4 GPUs] Run base model (Qwen2.5-7B) zero-shot evaluation on 
      ALL test benchmarks (in-domain + OOD)
      -> establishes difficulty buckets
  [ ] [16 GPUs] Start SFT Tier 1 runs: 6 domains * 5K data
      -> 4 GPUs each, 6 parallel -> all finish by noon

Afternoon (12:00-20:00):
  [ ] [24 GPUs] SFT Tier 2 runs: 3 domains * 3 data sizes = 9 runs
      -> 4 GPUs each, 6 parallel -> 2 batches -> done by 16:00
  [ ] [24 GPUs] DPO data construction: generate rollouts for DPO 
      preference pairs from base model
  [ ] [24 GPUs] Start DPO Tier 1 runs: 6 domains * 5K
      -> 4 GPUs each, 6 parallel -> done by 20:00

Evening (20:00-24:00):
  [ ] Evaluate all completed SFT models (in-domain + OOD)
  [ ] Evaluate all completed DPO models
  [ ] Check SFT results -- early signal on H1 (domain effect)

GPU-hours Day 1: ~200
Completed: 15 SFT runs + 6 DPO runs + base eval + CoT data

===================================================================
  DAY 2: All GRPO Training (Tier 1 + Tier 2)
===================================================================

Morning -> Evening (full day, runs are longer):
  [ ] Batch 1 (8:00-16:00): 3 GRPO Tier 1 runs (Math, Code, Science)
      -> 8 GPUs each, 3 parallel -> ~8 hours
  [ ] Batch 2 (16:00-24:00): 3 GRPO Tier 1 runs
      (Medicine, Law, Commonsense)
      -> 8 GPUs each, 3 parallel -> ~8 hours

Overnight (runs continue):
  [ ] Start GRPO Tier 2 (data-size ablation) on freed GPUs as Tier 1
      completes

Note: GRPO at N=100 and N=500 are fast (~1-2 hours), so we can 
squeeze these in between Tier 1 batches.

GPU-hours Day 2: ~450
Completed: 6 GRPO Tier 1 + begin Tier 2

===================================================================
  DAY 3: GRPO Tier 2 Completion + Hybrid Runs + Evaluation
===================================================================

Morning (8:00-14:00):
  [ ] Complete remaining GRPO Tier 2 runs (N=500, 2K for 3 domains)
      -> 6 runs * ~3 hours avg = 2 batches of 3 -> done by 14:00
  [ ] Begin evaluating all GRPO models (batch eval with vLLM)

Afternoon (14:00-22:00):
  [ ] Tier 3: Hybrid runs (SFT->GRPO, GRPO->SFT) for 3 domains
      -> 6 runs on 8 GPUs each -> 2 batches -> done by 22:00
      -> SFT->GRPO: load best SFT checkpoint, continue with GRPO
      -> GRPO->SFT: load best GRPO checkpoint, continue with SFT

Evening:
  [ ] Evaluate all hybrid models
  [ ] Compile preliminary results table
  [ ] Checkpoint: do we see the expected domain dichotomy? If not, 
      diagnose (check reward statistics, training loss curves)

GPU-hours Day 3: ~400
Completed: All Tier 1-3 + all evaluations

===================================================================
  DAY 4: Scaling Experiments + Deep Analysis
===================================================================

Morning (8:00-14:00):
  [ ] [OPTIONAL] Tier 4: Scaling runs -- Qwen2.5-3B and Qwen2.5-14B
      -> 4 runs on 3B (4 GPUs each) + 4 runs on 14B (8-12 GPUs each)
      -> Priority: Math + Medicine for both sizes
  [ ] [PARALLEL] Deep analysis on completed results:
      - Compute difficulty stratification (Easy/Med/Hard per domain)
      - Generate all tables and figures
      - Compute bootstrap confidence intervals
      - Analyze reasoning chains (length, error types)

Afternoon (14:00-20:00):
  [ ] Complete scaling evaluations
  [ ] Reward signal analysis: plot reward distribution per domain
      for GRPO
  [ ] Training dynamics analysis: accuracy vs training step curves
  [ ] Generate the "RLVR Benefit Frontier" heatmaps (Figure 5)
  [ ] Draft key findings list

Evening (20:00-24:00):
  [ ] Finalize all result tables
  [ ] Generate publication-quality figures
  [ ] Identify any surprising or contradictory findings

GPU-hours Day 4: ~300
Completed: All experiments done

===================================================================
  DAY 5: Paper Writing + Final Figures
===================================================================

Full day dedicated to writing:
  [ ] Morning: Introduction, Related Work, Method sections
  [ ] Afternoon: Results, Analysis sections
  [ ] Evening: Discussion, Conclusion, Abstract
  [ ] Polish figures to publication quality
  [ ] Compile LaTeX, check formatting
  [ ] Proofread

GPU usage: Minimal (only for any last-minute re-evaluations)
```

### 9.2 Compute Budget

| Category | Runs | GPUs/Run | Hours/Run (avg) | GPU-hours |
|----------|------|----------|----------------|-----------|
| **SFT Tier 1** (6 domains * 5K) | 6 | 4 | 3 | 72 |
| **SFT Tier 2** (3 domains * 3 sizes) | 9 | 4 | 1.5 | 54 |
| **DPO Tier 1** (6 domains * 5K) | 6 | 4 | 4 | 96 |
| **GRPO Tier 1** (6 domains * 5K) | 6 | 8 | 8 | 384 |
| **GRPO Tier 2** (3 domains * 3 sizes) | 9 | 8 | 4 | 288 |
| **Hybrid Tier 3** (3 domains * 2) | 6 | 8 | 6 | 288 |
| **Scaling Tier 4** (2 * 2 * 2) | 8 | 6 | 5 | 240 |
| **SFT CoT data generation** (72B) | 1 | 4 | 6 | 24 |
| **Evaluation** (all models * all benchmarks) | -- | 4 | 20 | 80 |
| **Base model eval + DPO data gen** | -- | 8 | 8 | 64 |
| **Total** | | | | **~1,590** |

**Available**: 24 GPUs * 24 hours * 5 days = 2,880 GPU-hours -> **55% utilization**, comfortable buffer.

### 9.3 API Budget

| Use | Calls | Cost |
|-----|-------|------|
| CoT data generation (backup, if 72B local fails) | 120K | ~$25 |
| GPT-4o evaluation (reasoning quality audit, 500 ex) | 500 | ~$10 |
| **Reserve** | -- | $965 |
| **Total** | | **~$35 used** |

Most of the budget is untouched -- our design is entirely local-compute.

---

## 10. Risk Analysis and Contingencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **veRL GRPO fails to converge on MCQ domains** | Medium | HIGH | Use OpenRLHF as fallback; simplify to REINFORCE if needed; verify on toy task first (Day 0) |
| **RLVR always wins (or always loses)** | Low-Med | HIGH (boring) | The hybrid and data-size ablations will still provide interesting findings; reframe as "RLVR is dominant/dominated -- here's why" |
| **No interesting crossover patterns** | Low | HIGH | Even if one method dominates, the OOD and difficulty analyses will show differential behavior |
| **GRPO training too slow (can't finish Tier 1 in Day 2)** | Low | Medium | Reduce group size G from 8->4; shorten max_new_tokens; drop commonsense domain (least interesting) |
| **SFT CoT data generation quality is poor** | Medium | Medium | Use GPT-4o-mini API as backup ($25); or use the base model's correct rollouts (from DPO data gen) |
| **LegalBench data is too small for data-size ablation** | Medium | Low | Replace law with commonsense in the data-size ablation; keep law only in Tier 1 |
| **Code execution environment issues** | Medium | Low | Use isolated Docker containers; fall back to regex-based code evaluation if sandboxing fails |
| **Qwen2.5-14B OOM on 8 GPUs** | Low | Low | Use 12 GPUs per 14B run or enable quantization (AWQ/GPTQ) |

### Contingency Priorities (if running behind schedule)

1. **Drop Tier 4** (scaling) -- nice-to-have but not essential
2. **Drop DPO** from Tier 1 -- focus story on RLVR vs SFT only
3. **Reduce Tier 2** from 3 domains to 2 (keep math + medicine as extremes)
4. **Drop one hybrid** (keep only SFT->GRPO, drop GRPO->SFT)
5. **NEVER drop**: Tier 1 (core comparison) or the OOD evaluation (key contribution)

---

## 11. Paper Outline

### Title
"When Does RLVR Beat SFT? A Controlled Multi-Domain Study of Reinforcement Learning vs Supervised Fine-Tuning for LLM Reasoning"

### Abstract (~250 words)

**[Motivation]** The post-DeepSeek-R1 era has seen widespread adoption of Reinforcement Learning with Verifiable Rewards (RLVR) for improving LLM reasoning. Yet practitioners lack guidance on a fundamental question: when should they invest in RLVR versus standard Supervised Fine-Tuning (SFT)?

**[Gap]** Existing comparisons are confined to single domains, use different base models, or conflate training paradigm differences with data quality differences.

**[Study]** We present the first controlled, multi-domain comparison of RLVR (GRPO), SFT, and DPO using the same base model (Qwen2.5-7B), same training data, and same evaluation protocol across six reasoning domains: mathematics, code, science, medicine, law, and commonsense reasoning. We systematically vary data size (100 to 10K), measure in-distribution and out-of-distribution performance, and stratify by task difficulty.

**[Findings]** We identify the "RLVR Benefit Frontier" -- the boundary conditions where RLVR transitions from advantageous to disadvantageous relative to SFT. Key findings: (1) RLVR outperforms SFT on procedural reasoning (math +5%, code +5%) but underperforms on knowledge-intensive tasks (medicine -3%, law -4%); (2) RLVR is more sample-efficient at small data scales but the advantage diminishes with more data; (3) RLVR-trained models consistently generalize better out-of-distribution; (4) sequential SFT->RLVR outperforms both individual methods across all domains.

**[Contribution]** We release a practical decision framework for when to use RLVR vs SFT, along with all training configurations and evaluation data to support future research.

### Paper Structure (8 pages)

```
S1. Introduction                           1.5 pages
  - Hook: "Everyone uses RLVR, but should they?"
  - The proliferation of RLVR and the missing comparison
  - Our study: controlled, multi-domain, multi-axis
  - Key findings preview
  - Contributions (3 bullets)

S2. Related Work                           1.0 page
  S2.1 RLVR for LLM Reasoning
  S2.2 SFT vs RL Comparisons (sparse, single-domain)
  S2.3 Curriculum and Data Selection

S3. Experimental Design                   1.5 pages
  S3.1 Setup: Base Model, Domains, Benchmarks
  S3.2 Training Methods: GRPO, SFT, DPO, Hybrids
  S3.3 Fairness Protocol: Same data, same model, same eval
  S3.4 Axes of Comparison: Domain, Difficulty, Data Size, OOD
  [Table: Domain and benchmark overview]
  [Figure 1: Experimental design overview]

S4. Results                                2.5 pages
  S4.1 Main Comparison Across Domains          [Table 1, Figure 1]
  S4.2 Data Efficiency: The Crossover Point    [Figure 2]
  S4.3 OOD Generalization Gap                  [Table 2, Figure 3]
  S4.4 Difficulty Stratification               [Figure 4]
  S4.5 The RLVR Benefit Frontier               [Figure 5]  <-- KEY
  S4.6 Hybrid Methods: SFT->RLVR              [Table 3]
  S4.7 Compute Efficiency                      [Table 4]

S5. Analysis and Discussion                1.0 page
  S5.1 Why Does RLVR Win on Procedural Tasks?
    - Exploration vs imitation
    - Reward signal sufficiency (base model accuracy > 20%)
  S5.2 Why Does SFT Win on Knowledge-Intensive Tasks?
    - SFT directly injects knowledge through demonstrations
    - RLVR can only reinforce existing capabilities
      ("The Invisible Leash" confirmed)
  S5.3 Why Is the Hybrid Best?
    - SFT bootstraps knowledge; RLVR refines reasoning
  S5.4 Practical Decision Framework
    [Figure: Decision tree for practitioners]
  S5.5 Limitations
    - Single base model family (Qwen2.5)
    - Binary rewards only (no process supervision)
    - MCQ format may favor certain methods

S6. Conclusion                             0.5 page
  - Summary of the RLVR Benefit Frontier
  - Practical guidelines
  - Future work: process rewards, open-ended tasks, more model families

References                                 ~2 pages (not counted)

Appendix (not counted):
  A. Training hyperparameters (full detail)
  B. Evaluation prompt templates
  C. Additional results: per-benchmark breakdown
  D. Training dynamics plots (loss/accuracy curves)
  E. Reasoning chain analysis examples
  F. Compute cost breakdown
```

### Key References to Cite and Position Against

```
% Core RL methods
DeepSeek-R1 (2501.12948)      - GRPO/RLVR paradigm origin
DAPO (2503.14476)              - GRPO improvements
GRPO/DeepSeekMath (2402.03300) - original GRPO paper

% Key competitors (must differentiate)
Med-RLVR (2502.19655)         - RLVR vs SFT in medicine (single domain)
The Invisible Leash (2507.14843) - theoretical RLVR limitations
SRL (ICLR 2026)               - hybrid SFT+RL (proposes method)
ReLIFT                        - dynamic switching (engineering)
Tsinghua Critique (2504.13837) - RLVR limitations

% Broader RLVR landscape
Delay, Plateau, Collapse (2605.02909)   - verifier noise effects
Crossing the Reward Bridge (2503.23829) - RLVR to diverse domains
RLVeR / Rate or Fate (2601.04411)       - noisy reward theory

% Datasets
MATH (Hendrycks et al.)
GSM8K (Cobbe et al.)
MedQA (Jin et al.)
LegalBench (Guha et al.)
ARC-Challenge (Clark et al.)
HellaSwag (Zellers et al.)
ScienceQA (Lu et al.)
MBPP (Austin et al.)
HumanEval (Chen et al.)
```

---

## 12. Differentiation from Competitors (Reviewer-Ready)

### vs. Med-RLVR (2502.19655)
> Med-RLVR compares RLVR and SFT in the medical domain only, finding RLVR matches SFT in-distribution and excels OOD. Our work extends this to 6 domains and reveals that this OOD advantage is universal, but the in-distribution comparison is domain-dependent -- RLVR loses to SFT on knowledge-intensive tasks.

### vs. "The Invisible Leash" (2507.14843)
> The Invisible Leash provides theoretical analysis of RLVR's limitations, showing it cannot escape the base model's support. We provide the empirical counterpart: when does this theoretical limitation actually matter in practice? Answer: on knowledge-intensive tasks where the base model lacks domain knowledge.

### vs. SRL (ICLR 2026)
> SRL proposes a specific hybrid SFT+RL method. We do not propose a new method -- instead, we characterize WHEN each existing method (SFT, RLVR, DPO, and simple sequential combinations) is the right choice. Our findings explain WHY hybrids like SRL work.

### vs. ReLIFT
> ReLIFT proposes dynamic switching between SFT and RL based on training signal. This is an engineering solution. We provide the systematic understanding that motivates such approaches, and show that even a simple sequential SFT->RLVR achieves strong results.

### vs. Tsinghua Critique (2504.13837)
> The Tsinghua critique argues that RLVR's reasoning capability is "locked in" by the base model. We provide nuanced empirical evidence: this is true for knowledge-intensive domains (base model's factual knowledge limits RLVR), but on procedural domains, RLVR can genuinely improve reasoning strategies within the base model's support.

---

## 13. Reproducibility and Release Plan

### Code
- Training scripts for all three methods (SFT, GRPO, DPO)
- Evaluation scripts with answer extraction per domain
- Data preparation scripts (subsampling, CoT generation)
- Analysis and figure generation notebooks

### Data
- Formatted training/test splits for all 6 domains at all data sizes
- Generated CoT demonstrations for SFT
- DPO preference datasets

### Models (optional)
- All trained checkpoints (~50 models, ~7GB each)
- Or: selected checkpoints for key conditions

### Results
- Raw evaluation outputs for all conditions
- Aggregate results tables in CSV format

---

## 14. Why EMNLP 2026?

1. **Empirical NLP focus**: EMNLP values comprehensive empirical studies and analyses
2. **Timeliness**: RLVR is the dominant paradigm in 2025-2026; the community urgently needs this comparison
3. **Practical impact**: Every NLP practitioner choosing post-training methods benefits from our findings
4. **Not a methods paper**: We don't propose a new method (which might face "incremental" criticism), but provide a fundamental characterization (which EMNLP values)
5. **Cross-domain scope**: Covers NLP-adjacent domains (science, medicine, law) that are core to EMNLP's scope

---

## Summary Checklist

- [x] Problem anchor: clear practitioner need
- [x] 6 domains with specific benchmarks and data sources
- [x] 3 training methods (SFT, GRPO, DPO) + 2 hybrids
- [x] Fairness protocol (same data, same model, same eval)
- [x] Data-size ablation (100, 500, 2K, 5K)
- [x] In-domain + OOD evaluation per domain
- [x] Difficulty stratification (Easy/Medium/Hard)
- [x] Detailed GRPO/SFT/DPO configurations
- [x] SFT training data construction pipeline
- [x] Expected findings with concrete numbers
- [x] Visualization plan (6 figures + 4 tables)
- [x] 5-day timeline with hour-by-hour GPU allocation
- [x] Compute budget: ~1,590 GPU-hours (55% of capacity)
- [x] API budget: ~$35 of $1,000
- [x] Risk analysis with contingencies
- [x] Paper outline with page allocation
- [x] Reviewer-ready differentiation from 5 key competitors
- [x] Reproducibility plan
