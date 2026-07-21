# Paper 2 Ideas: GPU-Training Direction for EMNLP/ACL 2026

**Date**: 2026-05-16
**Constraints**: 24×H200 GPU, ~$500 API, 5 days, complementary to Paper 1 (code agent failure analysis + behavioral scaffolding)

---

## Step 1: Brainstormed Directions (14 ideas)

### From User Suggestions

1. **Agent Token Efficiency via RL** — Train agents to use fewer tokens per SWE-bench problem (100K+ → much less). RL with token penalty.
2. **Recover vs Restart Meta-Decision** — Train a meta-controller that decides "continue recovering from this error state" vs "restart from scratch".
3. **Small Model Code Agent via RL** — RL-train a 4B model (mini-coder) from SFT-only 26.8% → 35-40% on SWE-bench.
4. **Curriculum for Agent RL** — Difficulty-aware task ordering for SWE-bench RL training (DeepSWE trains all problems uniformly).
5. **Cross-Task Agent Transfer** — Train on WebShop/ALFWorld → evaluate transfer to SWE-bench.

### Additional Directions

6. **SFT→RL Uplift Study for Code Agents** — Systematic "SFT Memorizes, RL Generalizes" study specifically for agent tasks at multiple scales (4B, 8B, 14B, 32B).
7. **Agent PRM for Code Tasks** — Train a process reward model specialized for code agent trajectories; use for test-time search and RL training.
8. **Agent Trajectory Distillation + RL** — Two-stage: first distill from strong agent, then RL fine-tune. Study interaction between SFT warmup quality and RL gain.
9. **Reward Shaping for Agent RL** — Dense intermediate rewards (did it find the right file? correct function? syntactically valid edit?) instead of sparse pass/fail.
10. **Multi-Agent Cooperative RL Training** — Train two agents (navigator + editor) cooperatively via RL (AT-GRPO style).
11. **RL-Trained Agent Generalization Study** — Do RL-trained SWE agents generalize to unseen repos/languages, or do they overfit to training distribution?
12. **Agent RL Reward Hacking Analysis** — Systematic study of how RL-trained code agents game test suites, write minimal patches, or exploit environment shortcuts.
13. **Reasoning Mode Selection for Agent RL** — When should agents think deeply vs call tools? Train to adaptively choose (relates to DemyAgent findings).
14. **Agent RL from Scratch vs SFT+RL** — Clean comparison: pure RL (DeepSWE-style) vs SFT warmup + RL at small scale (4B-8B). Which is better and why?

---

## Step 2: Competition Landscape (2025-2026)

### Idea 1: Agent Token Efficiency via RL
**Competition**: HIGH
- **Agent-Omit** (ICML 2026): Already trains agents to omit redundant thoughts/observations via RL. Core contribution directly overlaps.
- **SWE-TRACE** (2026.04): Trajectory compression + token-efficient training + PRM-guided search. Comprehensive framework.
- Both are published at top venues in 2026.
**Verdict**: ❌ **Novelty 3/10** — Agent-Omit is essentially this idea. We'd be a late follower.

### Idea 2: Recover vs Restart Meta-Decision
**Competition**: LOW-MEDIUM
- **Agent-R** (ByteDance, 2025.01): Constructs recovery data via MCTS, but doesn't study the meta-decision.
- **ACRFence** (2026.03): Studies checkpoint-restore security, not the decision of when to use it.
- No paper directly trains a "recover vs restart" classifier/policy.
**Verdict**: ⚠️ **Novelty 6/10** — Interesting but thin as a standalone paper. Hard to get 8 pages of content. Could be one experiment in a larger paper.

### Idea 3: Small Model Code Agent via RL (★★★ TOP CANDIDATE)
**Competition**: LOW
- **mini-coder** (2026.05.02): 4B model, SFT-only (distillation from 30B), 26.8% on SWE-bench. **Explicitly did NOT try RL**. Authors say RL is "promising future direction."
- **DeepSWE** (2025.07): Pure RL, but only 32B model. **No small model RL**.
- **SkyRL** (2025.05): Trained 7B/8B/14B with RL, but results are weak (7B: 14.6%, 8B: 9.4%). They used only ~300 training examples and basic GRPO.
- **SWE-RL** (2025.12): Self-play RL, only large models.
- **Open-AgentRL/DemyAgent** (ICML 2026): Trained 4B model but on math/code-generation, NOT SWE-bench agent tasks.
- **code-agent-grpo** (personal project, 2026.05): 7B GRPO for SWE-bench, no results published, 0 stars.
- **Key gap**: Nobody has done serious RL training (GRPO/ARPO) on 4B-8B models for SWE-bench with proper data (SWE-smith 50K), proper curriculum, and proper scale (24×H200). mini-coder shows SFT ceiling at 26.8% — **can RL break through?**
**Verdict**: ✅ **Novelty 7.5/10** — Clean gap. Mini-coder authors explicitly punt on RL. SkyRL's small model results are weak due to tiny data. We can do this properly.

### Idea 4: Curriculum for Agent RL (★★ STRONG CANDIDATE)
**Competition**: MEDIUM
- **SPEED-RL** (2025.06): Curriculum for reasoning RL (math), not agent tasks. Key idea: train on "intermediate difficulty."
- **CURATE** (ICML 2025): Automatic curriculum for RL agents, but game/navigation environments, not code.
- **SSR/Self-Play SWE-RL** (2025.12): Uses self-play (inject → repair) with increasing difficulty, but it's self-play curriculum, not task selection curriculum.
- **DeepSWE**: All problems equal weight. No curriculum.
- **No paper applies difficulty-aware curriculum to SWE-bench agent RL training**.
**Verdict**: ✅ **Novelty 7/10** — SPEED-RL proved curriculum works for reasoning RL. Transferring to agent RL on SWE-bench is natural but untested. Risk: effect may be small.

### Idea 5: Cross-Task Agent Transfer
**Competition**: LOW
- AgentBench (2023) evaluates across tasks but doesn't study transfer.
- No systematic study of training on Task A → evaluating on Task B for RL-trained agents.
**Verdict**: ⚠️ **Novelty 7/10 but HIGH risk** — If transfer is zero (likely), paper is negative-result only. If transfer works, it's great. Too risky for 5-day sprint.

### Idea 6: SFT→RL Uplift Study for Agents
**Competition**: MEDIUM-HIGH
- "SFT Memorizes, RL Generalizes" (2025.01) — already done for text/visual environments.
- Open-AgentRL/DemyAgent (ICML 2026) — studies data/algo/reasoning for agent RL but doesn't focus on SFT→RL comparison.
**Verdict**: ⚠️ **Novelty 5/10** — Extending known finding to new domain. Incremental.

### Idea 7: Agent PRM for Code Tasks
**Competition**: HIGH
- **AgentPRM** (2025.02/2025.11) — directly proposes PRM for agent tasks.
- **SWE-TRACE** (2026.04) — uses Rubric-PRM for code agent trajectories.
- **StepAgent** (2025) — step-level reward for agent optimization.
**Verdict**: ❌ **Novelty 3/10** — AgentPRM + SWE-TRACE already cover this.

### Idea 8: Agent Trajectory Distillation + RL
**Competition**: MEDIUM-HIGH
- **Structured Agent Distillation** (2025.05) — distills agent trajectories.
- **Agent Distillation** (NeurIPS 2025) — distills LLM agent behavior into small models.
- **mini-coder** — is exactly SFT distillation for SWE agents.
**Verdict**: ⚠️ **Novelty 5/10** — Combining SFT distillation + RL is natural extension but not novel framing.

### Idea 9: Reward Shaping for Agent RL
**Competition**: MEDIUM
- **SWE-TRACE** (2026.04) uses rubric-based dense rewards.
- **code-agent-grpo** uses multi-component reward (format + apply + test).
- General reward shaping is well-studied.
**Verdict**: ⚠️ **Novelty 5/10** — SWE-TRACE's rubric rewards partially cover this.

### Idea 10: Multi-Agent Cooperative RL
**Competition**: MEDIUM
- **PettingLLMs/AT-GRPO** (ICLR 2026) — multi-agent GRPO training.
- Not applied to SWE-bench specifically.
**Verdict**: ⚠️ **Novelty 6/10** — Interesting but engineering-heavy, hard in 5 days.

### Idea 11: RL-Trained Agent Generalization
**Competition**: LOW
- "SFT Memorizes, RL Generalizes" shows RL generalizes better in simple envs.
- Nobody tested this for SWE-bench: does RL agent generalize to new repos/frameworks?
**Verdict**: ✅ **Novelty 7/10** — Good question but purely empirical. Could merge with Idea 3.

### Idea 12: Agent RL Reward Hacking
**Competition**: LOW for code agents specifically
- General reward hacking extensively studied.
- No paper specifically documents how SWE-bench RL agents game tests.
**Verdict**: ⚠️ **Novelty 6/10** — Interesting but hard to control for; findings may be dataset-specific.

### Idea 13: Reasoning Mode Selection
**Competition**: HIGH
- **DemyAgent** (ICML 2026) already found "deliberative reasoning with selective tool calling" is optimal.
**Verdict**: ❌ **Novelty 3/10** — DemyAgent covers this.

### Idea 14: RL from Scratch vs SFT+RL (★ MERGE CANDIDATE)
**Competition**: LOW for agent tasks
- DeepSWE showed pure RL works at 32B. Nobody compared SFT+RL vs pure RL at small scale.
**Verdict**: ✅ **Novelty 7/10** — Natural sub-experiment of Idea 3.

---

## Step 3: Top 3 Candidates

### 🥇 #1: Small Model Code Agent via RL (Idea 3 + 4 + 11 + 14 merged)

**Working Title**: "Scaling Down, Not Up: RL Training for Small Code Agents"

**Core Insight**: Everyone is scaling UP (32B, 70B) for agent RL. But can RL unlock capabilities in small (4B-8B) models that SFT distillation cannot? mini-coder shows SFT saturates at 26.8% for 4B. Can RL push past this ceiling? And HOW should you train small models differently from large ones?

**Why novel**:
- mini-coder (4B, SFT-only): 26.8%. Explicitly didn't try RL.
- SkyRL (7B, RL): only 14.6% with 300 examples — clearly undertrained.
- DeepSWE (32B, RL): 42.2%. Nobody tried this recipe on small models.
- Open-AgentRL/DemyAgent: 4B but math/code-gen, not SWE-bench agent.
- **The gap is real and clearly acknowledged by mini-coder authors**.

**Novelty**: 8/10

### 🥈 #2: Curriculum Learning for Agent RL (Idea 4, standalone)

**Working Title**: "Not All Bugs Are Created Equal: Curriculum Learning for Code Agent RL"

**Core Insight**: SPEED-RL showed that training on intermediate-difficulty problems accelerates reasoning RL. DeepSWE trains on all SWE problems equally. For code agents, difficulty has multiple dimensions: bug type, codebase complexity, required actions. A curriculum should help.

**Why novel**:
- SPEED-RL (2025): curriculum for math reasoning RL, not agent.
- CURATE (ICML 2025): curriculum for game agents, not code.
- DeepSWE/SkyRL: no curriculum.
- Nobody has defined or measured "difficulty" for SWE-bench RL problems.

**Novelty**: 7/10

### 🥉 #3: Recover vs Restart Meta-Decision (Idea 2)

**Working Title**: "Should I Stay or Should I Go? Learning When to Restart vs Recover in Code Agents"

**Core Insight**: Current agents either always try to fix forward or restart randomly. A trained meta-controller could save tokens and improve success rate.

**Why novel**:
- Agent-R constructs recovery data but doesn't address the restart decision.
- No paper trains this specific meta-decision.

**Novelty**: 6/10 — Thin for standalone paper but strong as a component.

---

## Step 4: Detailed Plan for Top Pick

# 🥇 "Scaling Down, Not Up: RL Training for Small Code Agents"

## Core Insight

The code agent RL training paradigm (DeepSWE, SkyRL, SWE-RL) has focused exclusively on 32B+ models. Meanwhile, mini-coder shows that SFT distillation from a 30B teacher gets a 4B model to 26.8% on SWE-bench — impressive but limited by teacher quality. **Can RL break through the SFT ceiling for small models?** This is a fundamental question about the interaction between model capacity and training paradigm for agent tasks.

The "SFT Memorizes, RL Generalizes" result (ICLR 2025) predicts RL should help — but this was shown for simple environments, not complex multi-step agent tasks where small models may lack fundamental capacity. **We don't know if RL's generalization benefit survives at small scale for hard agent tasks.**

## Why This Is Interesting (Story Arc)

1. **Practical impact**: If RL can push 4B models to 35%+, this democratizes SWE agent research. Currently you need 32B+ (DeepSWE needs 64×H100).
2. **Scientific question**: Does the "RL > SFT" generalization gap hold for agent tasks? Or is there a capacity threshold below which RL can't help?
3. **Connects to EMNLP's "New Missions" theme**: Democratizing agent capabilities, responsible compute.
4. **Surprising if positive**: 4B model with RL matching or beating SFT-trained 30B models would be headline result.
5. **Informative if negative**: Understanding WHERE RL fails for small models (exploration? credit assignment? capacity?) is valuable.

## Method Design

### Phase 1: Baselines (Day 1 morning)
- Reproduce mini-coder-4B (SFT) performance with mini-swe-agent + SWE-bench Verified
- Also baseline: Qwen3-4B-Instruct (no fine-tuning), Qwen3-8B-Instruct

### Phase 2: RL Training (Day 1-3)
**Models**: Qwen3-4B, Qwen3-8B (also test 1.7B as lower bound, 14B as upper bound)

**Training Recipe**:
- **Framework**: rLLM (DeepSWE's framework) or SkyRL (VeRL + OpenHands)
- **Algorithm**: GRPO (proven for agent tasks), also try ARPO (ICLR 2026, designed for multi-turn agents)
- **Data**: R2E-Gym subset (4,578 instances) — proven by DeepSWE
- **Reward**: Binary (test pass/fail) + multi-component (patch format + apply + test_fail_to_pass + test_pass_to_pass)
- **Key hyperparameters for small models** (our contribution):
  - Shorter max trajectory length (small models degrade at long context)
  - Smaller action space (simplified tool set)
  - Higher KL penalty (prevent catastrophic drift on small models)
  - Gradient accumulation to compensate for smaller batch effective variance

**Curriculum variant** (ablation):
- Train first on "easy" problems (historically high solve rate across models), then gradually introduce harder ones
- Compare: uniform sampling vs curriculum vs anti-curriculum (hard-first)
- Difficulty metric: aggregate solve rate from mini-coder pass@100

### Phase 3: Analysis (Day 3-4)
Key experiments:
1. **SFT ceiling vs RL**: For each model size, compare best SFT (mini-coder recipe) vs best RL vs SFT+RL
2. **Scaling curve**: Plot resolve rate vs model size × training method. Does RL's advantage grow or shrink with scale?
3. **Generalization test**: Evaluate on SWE-bench Verified (not in R2E-Gym training set). Does RL generalize better than SFT?
4. **Failure analysis**: What problem types do RL-trained small models solve that SFT models can't? (Connects to Paper 1's failure taxonomy!)
5. **Token efficiency**: Do RL-trained models use fewer tokens per attempt? (Free analysis)
6. **Reward hacking check**: Do small models develop degenerate strategies (trivial patches, test manipulation)?

### Phase 4: Writing (Day 4-5)

## Precise Differentiation from Competitors

| Work | Focus | Our Difference |
|------|-------|---------------|
| DeepSWE (2025.07) | 32B pure RL, 64×H100 | We study 4B-8B; can small models benefit from RL too? |
| mini-coder (2026.05) | 4B SFT distillation | We add RL on top; does RL break SFT ceiling? |
| SkyRL (2025.05) | 7B-14B RL, ~300 examples | We use full-scale data (4500+); proper training |
| Open-AgentRL (ICML 2026) | 4B agent RL for math | We study code agent tasks, fundamentally different env |
| "SFT Memorizes, RL Generalizes" (2025) | Simple envs | We test if this holds for complex multi-step agent tasks |
| SPEED-RL (2025.06) | Curriculum for math RL | We adapt curriculum to agent RL (ablation) |
| Agent-Omit (ICML 2026) | Token efficiency via thought omission | Orthogonal; we study training paradigm, they study inference |

## Experiment Design

| Exp | Model | Method | Data | Hardware | Time |
|-----|-------|--------|------|----------|------|
| E1 | Qwen3-4B | SFT (mini-coder repro) | SWE-smith 50K | 4×H200 | 6h |
| E2 | Qwen3-4B | GRPO | R2E-Gym 4.5K | 8×H200 | 18h |
| E3 | Qwen3-4B | SFT+GRPO | SWE-smith→R2E-Gym | 8×H200 | 24h |
| E4 | Qwen3-8B | SFT (distill from 30B) | SWE-smith 50K | 8×H200 | 12h |
| E5 | Qwen3-8B | GRPO | R2E-Gym 4.5K | 16×H200 | 24h |
| E6 | Qwen3-8B | SFT+GRPO | SWE-smith→R2E-Gym | 16×H200 | 36h |
| E7 | Qwen3-4B | GRPO+Curriculum | R2E-Gym (sorted) | 8×H200 | 18h |
| E8 | Qwen3-4B | ARPO | R2E-Gym 4.5K | 8×H200 | 18h |
| E9 | Qwen3-14B | GRPO (reference) | R2E-Gym 4.5K | 24×H200 | 36h |
| Eval | All | SWE-bench Verified | 500 instances | 4×H200 | 4h each |

**Total GPU time**: ~5 days with 24×H200 (parallel runs possible)

## 24×H200 Training Config

```yaml
# For 4B model GRPO (fits on 8 GPUs)
model:
  base: Qwen/Qwen3-4B  # or mini-coder-4b for SFT+RL
  context_length: 16384  # shorter than 32B (they use 32K)
  
training:
  algorithm: GRPO
  num_gpus: 8
  batch_size: 32  # smaller than DeepSWE's 64 (smaller model)
  rollout_batch_size: 16
  num_rollouts_per_prompt: 8
  learning_rate: 5e-6  # slightly higher than 32B
  kl_coeff: 0.05  # higher KL to prevent drift
  max_steps: 2000
  
environment:
  framework: mini-swe-agent  # lightweight
  max_turns: 20  # vs DeepSWE's 50+ (small model degrades)
  tools: [bash, str_replace_editor]
  
data:
  train: R2E-Gym-Subset (4578 instances)
  eval: SWE-bench-Verified (500 instances)
```

## 5-Day Timeline

| Day | Task | GPUs Used | Output |
|-----|------|-----------|--------|
| **Day 1** | Setup rLLM/SkyRL env; start SFT baselines (E1, E4) | 12×H200 | SFT baselines running |
| **Day 2** | Start RL training (E2, E3, E5) in parallel | 24×H200 | RL training running |
| **Day 3** | Continue RL; start E7 (curriculum), E8 (ARPO); eval SFT models | 24×H200 | SFT results; RL checkpoints |
| **Day 4** | Eval all RL models; analysis experiments (generalization, failure analysis) | 16×H200 | All numbers |
| **Day 5** | Writing; figures; submission | Minimal | Paper draft |

## Expected Results and Fallback

**Optimistic scenario** (40% likely):
- RL pushes 4B from 26.8% → 33-38%, approaching SFT-32B performance
- Clear RL > SFT at all scales, but bigger gap at smaller scales
- Story: "RL is MORE important for small models, not less"

**Neutral scenario** (40% likely):
- RL gives modest improvement (26.8% → 30-32%)
- SFT+RL > pure RL > pure SFT for small models
- Story: "SFT provides good initialization, RL adds generalization"

**Pessimistic scenario** (20% likely):
- RL barely helps at 4B (< 2% gain), helps more at 8B+
- Story: "There's a capacity threshold for agent RL — below 8B, SFT distillation is the better use of compute"

**All three scenarios make for a good paper.** The pessimistic finding (capacity threshold for agent RL) would be equally publishable and practically important — it tells the community "don't waste compute RL-training small models, use SFT instead."

## Paper Structure (8 pages)

1. **Introduction** (1p): mini-coder shows SFT ceiling at 4B; DeepSWE shows RL works at 32B; does RL work at 4B?
2. **Background & Related Work** (1p): Agent RL landscape, SFT distillation, scaling laws
3. **Experimental Setup** (1.5p): Models, training recipes, evaluation
4. **Main Results** (1.5p): SFT vs RL vs SFT+RL across scales, scaling curves
5. **Analysis** (2p):
   - What does RL teach that SFT doesn't? (behavioral analysis)
   - Generalization to unseen repos
   - Curriculum effect
   - Token efficiency
   - Failure mode analysis (connects to Paper 1!)
6. **Discussion & Conclusion** (1p): Implications for democratizing agent research

## Key Figures/Tables

- **Fig 1**: Scaling curve — resolve rate vs model size, colored by training method (SFT/RL/SFT+RL)
- **Fig 2**: Training dynamics — RL reward curve for 4B vs 8B vs 14B (do small models plateau earlier?)
- **Table 1**: Main results — all models × methods × eval sets
- **Table 2**: Problem-type breakdown — which bug types benefit most from RL?
- **Fig 3**: Generalization gap — in-distribution vs out-of-distribution performance by training method
- **Table 3**: Curriculum ablation results

## Complementarity with Paper 1

Paper 1 studies WHAT goes wrong (failure taxonomy + behavioral scaffolding).
Paper 2 studies HOW to train agents better (RL for small models).
- Paper 1's failure taxonomy can explain Paper 2's results: which failure types does RL fix?
- Paper 2's models are good subjects for Paper 1's behavioral scaffolding experiments.
- Can submit both to same venue without conflict — different angles entirely.

## API Budget

- SFT training: $0 (all local GPU)
- RL training: $0 (all local GPU with mini-swe-agent + Docker)
- Evaluation: $0 (local inference with vLLM)
- Ceiling analysis with GPT-4.1: ~$50 (100 SWE-bench problems for reference)
- **Total API cost: ~$50** (mostly GPU-bound paper!)

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| rLLM/SkyRL setup takes too long | Medium | Day 1 entirely for setup; fallback to simpler VeRL |
| 4B model RL doesn't converge | Low | Train 8B in parallel; use as comparison |
| Docker/SWE-bench env issues | Medium | Use R2E-Gym pre-built environments |
| Results are trivial ("RL helps a bit") | Medium | Focus on analysis: WHY and WHEN does RL help |
| Can't reproduce mini-coder baseline | Low | Use their published checkpoint directly |

---

## Summary Recommendation

**Strong recommendation: Idea #1 (Small Model Code Agent via RL)**

Reasons:
1. **Clear gap**: mini-coder (May 2026) explicitly didn't try RL. The paper ASKS for this follow-up.
2. **GPU-heavy**: Perfect use of 24×H200. Almost zero API cost.
3. **Low result risk**: All outcomes (RL helps / RL doesn't help / capacity threshold) are publishable.
4. **Rich analysis**: Connects to Paper 1 (failure types), to scaling laws, to generalization.
5. **5-day feasible**: rLLM already exists; mini-coder provides baselines; R2E-Gym provides data.
6. **EMNLP fit**: Empirical + analysis + method (curriculum RL ablation).
7. **Complementary to Paper 1**: Different angle (training vs analysis), different contribution.

**Backup**: If setup proves impossible in Day 1, pivot to **Idea #2 (Curriculum for Agent RL)** using existing SkyRL framework on 8B model only — smaller scope but still novel.
