# Method Design: SelfCurriculum

## Self-Evolving Curriculum Learning for Domain-Specific LLM Reasoning

---

## 1. Overview

SelfCurriculum is a unified framework for autonomous LLM self-improvement through self-evolving curriculum reinforcement learning across diverse reasoning domains. The core idea: an LLM alternates between generating domain-specific reasoning problems at adaptive difficulty and solving them, using a **Composite Pseudo-Verifier (CPV)** to provide reward signals for GRPO training — all without external labeled data.

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SelfCurriculum Loop                    │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Challenger   │───>│  Problem Pool │                  │
│  │  (Generator)  │    │  (Per-Domain) │                  │
│  └──────────────┘    └──────┬───────┘                   │
│         ▲                   │                            │
│         │            ┌──────▼───────┐                   │
│   Difficulty         │    Solver     │                   │
│   Feedback           │  (Reasoner)   │                   │
│         │            └──────┬───────┘                   │
│         │                   │                            │
│  ┌──────┴───────┐    ┌──────▼───────┐                   │
│  │  Curriculum   │<───│  Composite   │                   │
│  │  Controller   │    │  Pseudo-     │                   │
│  └──────┬───────┘    │  Verifier    │                   │
│         │            └──────┬───────┘                   │
│         │                   │                            │
│  ┌──────▼───────────────────▼───────┐                   │
│  │       GRPO Training Update       │                   │
│  │  (Challenger + Solver jointly)   │                   │
│  └──────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 Domain-Conditioned Challenger (Problem Generator)

The Challenger generates domain-specific reasoning problems. Unlike R-Zero (unconstrained math generation), our Challenger is conditioned on:
- **Domain tag** $d \in \{science, law, medicine, math, general\}$
- **Difficulty target** $\delta \in [0, 1]$ (estimated via curriculum controller)
- **Format constraint** $f \in \{MCQ, short\text{-}answer, true\text{-}false\}$

**Generation Prompt Template:**
```
Generate a {format} {domain} reasoning problem at difficulty level
{difficulty}/10. The problem should test {reasoning_type} and have
a single correct, verifiable answer.

Domain: {domain}
Difficulty: {difficulty}
Format: {format}

[Few-shot exemplars from seed set]

Problem:
```

**Challenger Reward** (3 components):

1. **Uncertainty Reward** (from R-Zero):
$$r_{unc}(x) = 1 - 2|p̂(x; S_φ) - 0.5|$$
where $p̂(x; S_φ)$ is the Solver's empirical accuracy on problem $x$ over $K$ rollouts.

2. **Domain Faithfulness Reward**:
$$r_{domain}(x) = \mathbb{1}[\text{CPV confirms } x \text{ is a valid } d\text{-domain problem}]$$
We use the CPV itself to verify that the generated problem is answerable and domain-appropriate.

3. **Diversity Reward** (inspired by OpenSIR):
$$r_{div}(x) = 1 - \max_{x' \in \mathcal{M}} \text{sim}(e(x), e(x'))$$
where $\mathcal{M}$ is a memory bank of recently generated problems and $e(\cdot)$ is the LLM's hidden-state embedding.

**Combined Challenger Reward:**
$$R_C(x) = \alpha \cdot r_{unc}(x) + \beta \cdot r_{domain}(x) + \gamma \cdot r_{div}(x)$$
Default: $\alpha = 0.5, \beta = 0.3, \gamma = 0.2$.

### 2.2 Solver (Reasoning Model)

The Solver attempts to solve each generated problem using chain-of-thought reasoning.

**Solver Input:**
```
Solve the following {domain} problem step by step.

{problem_text}

Think carefully and provide your final answer.
```

**Solver Output:** Chain-of-thought reasoning + final answer.

**Solver Reward:**
$$R_S(x, y) = \text{CPV}(x, y)$$
where $y$ is the Solver's answer and CPV provides a [0, 1] reward signal.

### 2.3 Composite Pseudo-Verifier (CPV)

The CPV is the key innovation enabling domain-agnostic verification without external labels or trained verifiers.

#### Component 1: Self-Consistency Score (SC)

For a problem $x$, generate $K$ solutions from the Solver:
$$\{y_1, y_2, ..., y_K\} \sim \pi_\theta(\cdot | x)$$

The self-consistency score:
$$SC(x, y_i) = \frac{\sum_{j=1}^{K} \mathbb{1}[\text{equiv}(y_j, y_i)]}{K}$$

where $\text{equiv}(\cdot, \cdot)$ is an equivalence function:
- For MCQ: exact match of answer letter.
- For short-answer: normalized string match (after removing whitespace, lowercasing).
- For numerical: approximate equality within tolerance $\epsilon$.

#### Component 2: Cross-Model Agreement (CMA)

Use a held-out reference model $\pi_{ref}$ (e.g., a different-family model) to independently solve the same problem:
$$\{z_1, z_2, ..., z_M\} \sim \pi_{ref}(\cdot | x)$$

Majority answer from reference model: $z^* = \text{mode}(\{z_1, ..., z_M\})$

Cross-model agreement:
$$CMA(x, y_i) = \mathbb{1}[\text{equiv}(y_i, z^*)]$$

#### Component 3: Confidence-Weighted Fusion

The final CPV score dynamically weights SC and CMA based on domain-specific reliability:

$$CPV(x, y_i) = w_d \cdot SC(x, y_i) + (1 - w_d) \cdot CMA(x, y_i)$$

where $w_d$ is a per-domain weight estimated online:
$$w_d = \sigma\left(\frac{\text{SC-accuracy}_d - \text{CMA-accuracy}_d}{\tau}\right)$$

We estimate SC-accuracy and CMA-accuracy on a small held-out validation set per domain (50–100 examples with known ground truth).

**Practical Implementation:**
- $K = 16$ samples for self-consistency (from Solver).
- $M = 8$ samples for cross-model agreement (from reference model).
- Reference model: Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct (different family from training model to maximize complementarity).
- Refresh $w_d$ every training epoch using rolling accuracy estimates.

#### CPV Reward Signal Types

1. **Binary Reward**: $R = \mathbb{1}[CPV(x, y) > 0.5]$
2. **Soft Reward** (preferred for non-math domains):
   $$R = CPV(x, y) \in [0, 1]$$
3. **Advantage-Normalized Reward** (for GRPO):
   $$\hat{A}_i = \frac{R_i - \mu(\mathbf{R})}{\sigma(\mathbf{R}) + \epsilon}$$

### 2.4 Adaptive Curriculum Controller

The Curriculum Controller manages difficulty progression per domain.

#### Difficulty Estimation

For domain $d$, the current difficulty level is:
$$\delta_d^{(t)} = 1 - \frac{1}{|B_d|} \sum_{x \in B_d} \text{acc}_\theta(x)$$

where $B_d$ is the current problem batch for domain $d$ and $\text{acc}_\theta(x)$ is the Solver's accuracy.

#### Adaptive Scheduling Algorithm

```python
class CurriculumController:
    def __init__(self, domains, initial_difficulty=0.3):
        self.difficulty = {d: initial_difficulty for d in domains}
        self.target_accuracy = 0.5  # Sweet spot (from DOTS/R-Zero)
        self.alpha = 0.1  # Learning rate for difficulty adjustment
        
    def update(self, domain, batch_accuracy):
        """Adjust difficulty based on solver performance."""
        if batch_accuracy > self.target_accuracy + 0.1:
            # Too easy → increase difficulty
            self.difficulty[domain] = min(1.0, 
                self.difficulty[domain] + self.alpha)
        elif batch_accuracy < self.target_accuracy - 0.1:
            # Too hard → decrease difficulty
            self.difficulty[domain] = max(0.0, 
                self.difficulty[domain] - self.alpha)
        # else: in sweet spot, maintain
        
    def get_difficulty(self, domain):
        return self.difficulty[domain]
    
    def get_domain_distribution(self):
        """Allocate more training to domains with low accuracy."""
        inverse_acc = {d: 1 - self.difficulty[d] 
                       for d in self.difficulty}
        total = sum(inverse_acc.values())
        return {d: v / total for d, v in inverse_acc.items()}
```

#### Domain Mixing Strategy

At each training iteration:
1. Sample domains proportionally to their learning need (inverse of current performance).
2. Within each domain, generate problems at the current target difficulty.
3. Interleave domain batches to prevent catastrophic forgetting.

**Domain mixing ratio** (evolves during training):
$$p_d^{(t)} = \frac{\text{gap}_d^{(t)}}{\sum_{d'} \text{gap}_{d'}^{(t)}}$$
where $\text{gap}_d^{(t)} = \delta_d^{(t)} - \text{acc}_d^{(t)}$ measures the learning opportunity.

---

## 3. Training Algorithm

### 3.1 GRPO Training (Adapted from DeepSeek-R1)

For both Challenger and Solver, we use GRPO with the following modifications:

**Solver GRPO Objective:**
$$\mathcal{L}_S(\theta) = -\mathbb{E}_{x \sim C_\phi} \left[\frac{1}{G} \sum_{i=1}^{G} \min\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}, 1 \pm \epsilon\right) \hat{A}_i \right)\right]$$

where:
- $G = 16$ rollouts per problem (serves dual purpose: training samples + self-consistency estimation).
- Advantages $\hat{A}_i$ computed from CPV rewards.
- Clipping $\epsilon = 0.2$ (with DAPO-style asymmetric clipping: $\epsilon_{high}=0.28, \epsilon_{low}=0.18$).

**Challenger GRPO Objective:**
$$\mathcal{L}_C(\phi) = -\mathbb{E} \left[\frac{1}{G_C} \sum_{j=1}^{G_C} \min\left(\frac{\pi_\phi(x_j|p)}{\pi_{\phi_{old}}(x_j|p)} \hat{A}_j^C, \text{clip}(\cdot) \hat{A}_j^C \right)\right]$$

where $G_C = 8$ problems per domain prompt, and advantages are from the Challenger reward $R_C$.

### 3.2 Iterative Self-Evolving Loop

```
Algorithm: SelfCurriculum Training

Input: Base LLM π₀, domains D, seed exemplars E_d per domain,
       reference model π_ref, num_iterations T
Output: Improved LLM π_T

1. Initialize: π_θ ← π₀ (Solver), π_φ ← π₀ (Challenger)
2. Initialize curriculum controller C with domains D

3. For iteration t = 1 to T:
   a. PROBLEM GENERATION PHASE:
      For each domain d ∈ D:
        - δ_d ← C.get_difficulty(d)
        - Generate N_d problems using Challenger π_φ:
          P_d = {x_1, ..., x_{N_d}} conditioned on (d, δ_d, exemplars E_d)
        - Filter: Remove duplicates and invalid problems via CPV
   
   b. SOLVING PHASE:
      For each problem x in ∪_d P_d:
        - Generate G rollout solutions: {y_1,...,y_G} ~ π_θ(·|x)
        - Compute CPV rewards: R_i = CPV(x, y_i) for each y_i
        - Compute advantages: Â_i = (R_i - μ(R)) / (σ(R) + ε)
   
   c. GRPO UPDATE PHASE:
      - Update Solver: θ ← θ - η∇L_S(θ) for E_train epochs
      - Update Challenger: φ ← φ - η∇L_C(φ) for E_train epochs
   
   d. CURRICULUM UPDATE PHASE:
      For each domain d:
        - Compute batch accuracy: acc_d = mean(R for problems in P_d)
        - C.update(d, acc_d)
   
   e. MEMORY BANK UPDATE:
      - Add high-value problems (reward variance > threshold) to replay buffer
      - Prune mastered problems (acc > 0.95)
   
   f. CHECKPOINT:
      - Evaluate on held-out validation sets per domain
      - Save best checkpoint per domain

4. Return π_T
```

### 3.3 Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | Qwen2.5-7B / Qwen2.5-14B | Strong base for reasoning |
| Reference Model | Llama-3.1-8B-Instruct | Different family for CMA |
| Rollouts per problem (G) | 16 | Balance: quality of SC vs. compute |
| CMA samples (M) | 8 | Sufficient for reliable majority vote |
| Iterations (T) | 5 | Diminishing returns after 5 |
| Problems per iteration | 2,000 per domain | ~10K total per iteration |
| GRPO epochs per iteration | 2 | Avoid over-fitting to current batch |
| Learning rate | 1e-6 (cosine decay) | Standard for GRPO |
| KL penalty coefficient | 0.01 | Prevent divergence from base |
| Temperature (generation) | 0.7 (Solver), 1.0 (Challenger) | Diversity for Challenger |
| Seed exemplars per domain | 200 | Minimal domain signal |
| Replay buffer size | 5,000 per domain | Historical revisiting |

---

## 4. Domain Instantiation

### 4.1 Science Domain

**Seed Data**: 200 examples from ScienceQA (MCQ, spanning physics, chemistry, biology, earth science).

**Challenger Prompt Adaptation:**
```
Generate a multiple-choice science question testing {sub_domain}
reasoning at difficulty {difficulty}/10.
The question should require multi-step scientific reasoning.
Include 4 answer choices (A, B, C, D) with exactly one correct.

Example:
{few_shot_exemplar}
```

**Verification**: CPV with SC + CMA. Science MCQ is well-suited for pseudo-verification since answers are discrete.

### 4.2 Legal Domain

**Seed Data**: 200 examples from LegalBench (rule application, issue spotting, multi-hop reasoning).

**Challenger Prompt Adaptation:**
```
Generate a legal reasoning question testing {legal_skill} at 
difficulty {difficulty}/10. The question should present a scenario
and ask about legal rules, their application, or implications.
Format: {MCQ|True-False}

Example:
{few_shot_exemplar}
```

**Verification**: CPV with emphasis on CMA ($w_{law}$ likely favors CMA due to legal ambiguity). SC alone may be unreliable for legal reasoning.

### 4.3 Medical Domain

**Seed Data**: 200 examples from MedQA (USMLE-style MCQs).

**Challenger Prompt Adaptation:**
```
Generate a medical reasoning question testing {medical_topic} at
difficulty {difficulty}/10. The question should describe a clinical
scenario and require diagnostic or treatment reasoning.
Include 5 answer choices (A-E) with exactly one correct.

Example:
{few_shot_exemplar}
```

**Verification**: CPV with both SC and CMA. Medical MCQ is verifiable; the challenge is domain accuracy of generated problems.

### 4.4 Math Domain (Baseline Comparison)

**Seed Data**: 200 examples from MATH dataset.

**Challenger Prompt**: Standard R-Zero format.

**Verification**: Standard majority voting (well-established for math). Also compare with CPV to measure CPV overhead.

---

## 5. Infrastructure Design

### 5.1 Compute Requirements (24x H200 GPUs)

**GPU Allocation:**
- 16 GPUs: GRPO training (model + optimizer states)
- 4 GPUs: Solver rollout generation (vLLM inference)
- 2 GPUs: Challenger problem generation (vLLM inference)
- 2 GPUs: Reference model inference (CMA)

**Framework**: veRL (used by DAPO) or OpenRLHF for GRPO training + vLLM for fast inference.

**Estimated Timeline per Iteration:**
- Problem generation: ~1 hour (10K problems across domains)
- Rollout generation: ~3 hours (16 rollouts × 10K problems)
- GRPO training: ~2 hours (2 epochs)
- CMA verification: ~1 hour (8 samples × 10K problems on reference model)
- **Total per iteration: ~7 hours**
- **5 iterations: ~35 hours ≈ 1.5 days**

### 5.2 API Budget for Reference Model ($1K)

If using an API-based reference model (e.g., GPT-4o-mini) instead of local:
- 10K problems × 8 CMA samples × 5 iterations = 400K API calls
- At ~$0.15/1K input tokens + $0.60/1K output tokens (GPT-4o-mini)
- ~200 tokens input + 50 tokens output per call
- Cost: 400K × (200×0.15/1K + 50×0.60/1K) ≈ $24 per run
- **Well within budget**, allowing multiple experimental runs.

**Recommendation**: Use local Llama-3.1-8B for most CMA; save API budget for ablation with stronger reference models (GPT-4o-mini, Claude-3.5-Haiku).

### 5.3 Data Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Seed Data   │────>│ Challenger   │────>│ Problem Pool│
│ (200/domain)│     │ Generation   │     │ (filtered)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│ GRPO Update │<────│ Advantage    │<────│ Solver      │
│             │     │ Computation  │     │ Rollouts    │
└──────┬──────┘     └──────────────┘     └──────┬──────┘
       │                   ▲                     │
       │            ┌──────┴──────┐              │
       │            │ CPV Rewards │<─────────────┘
       │            │ SC + CMA    │
       │            └─────────────┘
       │
       ▼
  Next Iteration
```

---

## 6. Theoretical Justification

### 6.1 Why Adaptive Difficulty at 0.5?

Following DOTS (arXiv: 2506.05316, Theorem 1), the expected squared gradient norm is maximized when the success rate is 0.5:

$$\mathbb{E}[\|\nabla \mathcal{L}\|^2] \propto p(1-p)$$

which is maximized at $p = 0.5$. This justifies targeting 50% Solver accuracy as the optimal difficulty.

### 6.2 Why Cross-Model Agreement Helps

**Intuition**: Self-consistency captures aleatoric uncertainty (randomness in sampling), while cross-model agreement captures epistemic uncertainty (gaps in model knowledge). Combining both provides a more robust verification signal.

**Formal argument**: Let $e_{self}$ be the error rate of self-consistency verification and $e_{cross}$ be the error rate of cross-model agreement. If errors are independent (different model families → different error patterns):
$$e_{CPV} \leq e_{self} \cdot e_{cross} \ll \min(e_{self}, e_{cross})$$

In practice, errors are not fully independent, but using different model families maximizes complementarity.

### 6.3 Convergence Argument

Following E2H Reasoner's approximate policy iteration analysis, curriculum learning with adaptive difficulty selection converges to a better policy than random sampling, requiring fewer total samples:

$$V^{\pi_{curriculum}} - V^{\pi_{random}} \geq \Omega\left(\frac{1}{\sqrt{T}}\right)$$

where $T$ is the number of training iterations.
