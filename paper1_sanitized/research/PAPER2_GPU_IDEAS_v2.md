# Paper 2: GPU-Training Direction Analysis (v2)

**Date**: 2026-05-16
**Deadline**: EMNLP 2026 ARR May cycle (2026-05-25)
**Resources**: 24×H200 GPU, ~$1000 API budget (most allocated to Paper 1)
**Constraint**: Must have GPU training component, novelty ≥7/10, 5-day feasible

---

## Summary of All Directions Evaluated

| # | Direction | Novelty | Competition | Result Risk | GPU Need | Recommended? |
|---|-----------|---------|-------------|-------------|----------|--------------|
| A | Thinking Token Causal Attribution | 6/10 | CTS, DuP-PO, Critical Tokens, Token-Budget-Aware | Medium | Medium | ❌ |
| B | RL vs SFT Internal Representation | 5/10 | "RL Squeezes SFT Expands", Stanford L2 Probing, NeurIPS'25 Best Paper | High | High | ❌ |
| C | Agent Planning via RL | 6/10 | PilotRL, AgentGym-RL | High | Very High | ❌ |
| **D** | **Reasoning Behaviors as Training Signal** | **7.5/10** | **Cognitive Behaviors (Stanford), but 1 task only** | **Low** | **High** | **⭐ Top Pick** |
| E | Reward Hacking in Code Agents | 4/10 | RHB (ICML'26), Countdown-Code, Advantage Modification | Medium | High | ❌ |
| **F** | **Reasoning Style Causal Anatomy** | **8/10** | **No direct competitor** | **Very Low** | **High** | **⭐⭐ BEST** |

---

## Direction A: Thinking Token Causal Attribution

### Core Idea
Train a model to predict which thinking tokens in reasoning traces causally contribute to the final answer. A "Process Reward Model for Thinking" — token-level importance scoring within the `<think>` block.

### Competition Analysis (CROWDED)
1. **CTS** (Conditional Token Selection, arXiv:2505.17827, May 2025) — Already does conditional importance scoring to compress CoT tokens. Trained on Qwen2.5-14B. 9.1% accuracy improvement with 13.2% fewer tokens on GPQA.
2. **DuP-PO** ("Do Thinking Tokens Help or Trap?", arXiv:2506.23840, June 2025) — Identifies "thinking trap" phenomenon, proposes Dual Policy PO to control thinking tokens. Math benchmarks.
3. **Critical Tokens Matter** (ICML 2025, arXiv:2410.xxxxx) — Token-level contrastive estimation for critical tokens in reasoning. cDPO algorithm.
4. **Token-Budget-Aware LLM Reasoning** (ACL 2025 Findings) — Dynamic token budget adjustment.
5. **R2R** (NeurIPS 2025) — Token-level routing between reasoning models.

### Gap Remaining
- CTS does importance scoring but for compression, not for understanding WHAT makes a thinking token useful
- No causal ablation study: "If we remove token X from thinking trace, does the answer change?"
- But the space is VERY active — 5+ papers on this exact topic in 2025

### Novelty Score: 6/10
Too crowded. The "causal attribution within thinking" angle is a natural next step that multiple groups are already pursuing.

### Verdict: ❌ Not recommended — high competition risk

---

## Direction B: RL vs SFT Internal Representation Analysis

### Core Idea
Use probing classifiers and representation analysis to understand WHAT changes inside a model after RL vs SFT training. Why does RL generalize better? What representations shift?

### Competition Analysis (VERY CROWDED)
1. **"RL Squeezes, SFT Expands"** (arXiv:2509.21128, NeurIPS 2025) — Already did trajectory-level and step-level reasoning graph analysis. Found RL concentrates into small subset of steps, SFT distributes. Models: 1.5B/7B/14B on math.
2. **"Does RL Really Incentivize Reasoning?"** (arXiv:2504.13837, **NeurIPS 2025 Best Paper**) — Found RL doesn't create new reasoning patterns, just reorganizes existing ones. Pass@k analysis shows base model has higher coverage.
3. **Stanford CS224R Project** (May 2026) — "Understanding the Effect of RL on Internal Representation of LLMs" — Mechanistic analysis comparing GRPO-based RL and SFT on Qwen3-1.7B using L2 probing.
4. **"SFT Memorizes, RL Generalizes"** (arXiv:2501.17161, 2025) — Controlled study on GeneralPoints game. RL generalizes to rule variants, SFT memorizes.
5. **Guru / Reasoning360** (arXiv:2506.14965) — Cross-domain RL study, 92K examples, 6 domains. Found domain-dependent transfer patterns.

### Gap Remaining
- None of these do probing ON CODE AGENTS specifically (all math/game domains)
- But the conceptual question "what does RL change inside?" is comprehensively addressed

### Novelty Score: 5/10
The core question is answered by multiple high-profile papers. Applying it to code domain alone is not enough novelty.

### Verdict: ❌ Not recommended — NeurIPS'25 best paper + 4 other papers cover this

---

## Direction C: Agent Planning via RL (Controlled Experiment)

### Core Idea
Design tasks with varying planning depth (1-step to 10-step), train with RL vs SFT, measure if planning ability specifically improves.

### Competition Analysis
1. **PilotRL** (arXiv:2508.00344, NeurIPS 2025) — Global planning-guided progressive RL training for agents. Already frames the RL+planning question, proposes a specific solution.
2. **AgentGym-RL** (GitHub, Feb 2026) — Multi-turn interactive RL training framework for agents.
3. **Agentic RL survey** (Sep 2025) — Comprehensive coverage of RL for agents.

### Gap Remaining
- PilotRL proposes a SOLUTION to planning but doesn't do a controlled ANALYSIS of what RL learns about planning
- A "planning depth analysis" study could be interesting
- But designing controlled tasks with precise planning depth is extremely hard in 5 days

### Novelty Score: 6/10
The controlled experiment angle is somewhat novel, but task design + training + analysis in 5 days is very tight.

### Verdict: ❌ Not recommended — too high execution risk for 5-day timeline

---

## Direction D: Reasoning Behaviors as Training Signal — "Which Cognitive Behaviors Actually Help?"

### Core Idea
The Stanford "Cognitive Behaviors" paper (arXiv:2503.01307) identified 4 behaviors (verification, backtracking, subgoal setting, backward chaining) that enable RL self-improvement, but ONLY on the Countdown game with 3B models. 

**Our extension**: Systematically decompose reasoning behaviors in training data, test each behavior's causal contribution to DOWNSTREAM GENERALIZATION across multiple domains (math, code, science), multiple scales (1.5B-14B), and both SFT and RL training paradigms.

### Competition Analysis
1. **Cognitive Behaviors (Stanford, arXiv:2503.01307, Mar 2025)** — **CLOSEST competitor**. Found behaviors > answer correctness for RL readiness. But:
   - Only Countdown game (1 task)
   - Only 3B models (Qwen-2.5-3B vs Llama-3.2-3B)
   - Only studies RL readiness, not downstream generalization
   - Does NOT isolate individual behaviors' contribution
2. **Project Aletheia** (arXiv:2601.14290, Jan 2026) — Trains 7B model on traces with backtracking. Shows verification behavior emerges. But only constraint satisfaction tasks.
3. **"Reasoning Pattern Matters"** (arXiv:2510.12643, Oct 2025) — Studies reasoning PATTERNS vs content. But "pattern" = procedural strategy (e.g., solve equations step by step), NOT individual cognitive behaviors.
4. **DC-CoT** (arXiv:2505.18759) — Data-centric benchmark for CoT distillation. Studies granularity, format, teacher model. Does NOT isolate behavioral components.
5. **"Unveiling Key Factors for CoT Distillation"** (ACL 2025) — Studies granularity, format, teacher. Found format has "minimal effect on SLMs." But "format" = structural presentation, NOT behavioral content.

### What Makes This Different (Gap Analysis)
Stanford paper's limitation is PRECISELY what we exploit:
- They showed behaviors matter, but only on 1 toy task → **we test on 3+ real domains**
- They didn't isolate each behavior → **we do controlled ablation: train with/without each behavior**
- They only tested RL readiness → **we test both SFT and RL generalization**
- They used 3B → **we use 1.5B/7B/14B for scale analysis**

### Core Insight (Why Surprising)
"The same correct answer, wrapped in different reasoning behaviors, leads to dramatically different generalization." This is non-trivial because current practice treats all correct CoT traces as equivalent — DC-CoT and most distillation work focus on correctness and granularity, NOT behavioral content.

### Novelty Score: 7.5/10
Real gap: comprehensive causal anatomy of individual reasoning behaviors across domains/scales. Stanford paper opened the door but only peeked through.

### Verdict: ⭐ Strong candidate, but needs careful framing to differentiate from Stanford paper

---

## Direction E: Reward Hacking in Code Agents

### Core Idea
Study how RL-trained code agents learn to hack rewards (e.g., modifying tests to pass instead of fixing bugs).

### Competition Analysis (VERY CROWDED for 2026)
1. **RHB** (Reward Hacking Benchmark, arXiv:2605.02964, May 2026, **ICML 2026**) — 13 frontier models, 6 exploit categories. Found RL training increases reward hacking (DeepSeek-V3 0.6% → R1-Zero 13.9%).
2. **"When Reward Hacking Rebounds"** (arXiv:2604.01476, Apr 2026) — Three-phase rebound pattern. Proposes Advantage Modification to suppress hacking during GRPO.
3. **Countdown-Code** (arXiv:2603.07084, Mar 2026) — Testbed for studying reward hacking emergence during training.
4. **"Benchmarking Reward Hack Detection in Code"** (Semantic Scholar, Jan 2026) — Already benchmarks detection in code generation specifically.

### Novelty Score: 4/10
Way too late. ICML 2026 paper already comprehensive. 4 papers in 3 months on this exact topic.

### Verdict: ❌ Not recommended — completely occupied

---

## Direction F (NEW): Reasoning Style Causal Anatomy — "The Anatomy of Useful Reasoning"

### ⭐⭐ TOP RECOMMENDATION ⭐⭐

### Core Idea

**Question**: When we train a model on reasoning traces, which SPECIFIC COMPONENTS of the reasoning actually transfer to new capabilities?

Current practice: Generate correct reasoning traces → train on them. But reasoning traces contain a MIX of behaviors:
- **Decomposition** (breaking problem into subproblems)
- **Self-verification** (checking intermediate results)
- **Self-correction/Backtracking** (noticing and fixing errors)
- **Exploration** (trying alternative approaches)
- **Summarization** (consolidating intermediate findings)

**Nobody has done a clean factorial experiment**: take the SAME problems, create training data with controlled presence/absence of each behavior, train models, and measure which behaviors contribute to:
1. In-domain accuracy
2. Out-of-domain generalization
3. Cross-domain transfer (math → code, code → math)
4. Robustness to problem perturbation

### Why This Is Different from ALL Existing Work

| Paper | What They Study | What They DON'T Study |
|-------|----------------|----------------------|
| Stanford Cognitive Behaviors | 4 behaviors enable RL readiness on Countdown | Individual behavior contribution; multiple domains; SFT; generalization |
| RL Squeezes, SFT Expands | Trajectory-level graph topology | Which behavioral COMPONENTS cause the topology difference |
| DC-CoT | Data quantity/quality/teacher for CoT distillation | Behavioral composition of training traces |
| ACL'25 Key Factors | Granularity, format, teacher model | Behavioral content within same format |
| Reasoning Pattern Matters | Pattern = procedural strategy | Individual cognitive behavior isolation |
| CTS / DuP-PO | Which tokens to keep/remove | Why certain tokens matter (behavioral function) |

### The Surprising Core Finding (Hypothesis)
Based on scattered evidence from existing work, we hypothesize:
1. **Self-correction in training data HURTS in-domain but HELPS OOD** — models trained on traces with self-correction learn to doubt and recover, which costs in-domain accuracy but gains robustness
2. **Verification tokens are the most transferable behavior** — they teach domain-general "checking" that transfers across math/code
3. **Decomposition helps within domain but doesn't transfer** — task decomposition is domain-specific
4. **There's an optimal "behavioral diversity" in training data** — too much self-correction = overthinking, too little = brittle

Evidence supporting these hypotheses:
- DuP-PO found thinking tokens can "trap" models → suggests self-correction has a cost
- Stanford paper found behaviors > correctness → suggests behavioral content matters more than we thought
- "RL Squeezes" found RL concentrates reasoning → suggests RL selects for efficient behaviors
- NeurIPS'25 best paper found RL doesn't create new patterns → suggests the BEHAVIORS in training data are what really matter, RL just selects among them

### Experimental Design (5-Day Plan)

**Phase 1 (Day 1): Data Construction — Behavioral Decomposition**
- Take 5K math problems (GSM8K, MATH) + 2K code problems (HumanEval+, MBPP+)
- Generate reasoning traces from DeepSeek-R1/Qwen3-32B with full reasoning
- Use GPT-4.1 to ANNOTATE each trace with behavioral labels:
  - `[DECOMPOSE]` — subproblem breakdown
  - `[VERIFY]` — intermediate checking
  - `[CORRECT]` — error detection + fix
  - `[EXPLORE]` — alternative approach
  - `[SUMMARIZE]` — consolidation
- Create CONTROLLED variants by REMOVING specific behaviors:
  - Full trace (all behaviors)
  - No-verify (remove all verification segments)
  - No-correct (remove all self-correction)
  - No-explore (remove all exploration)
  - Verify-only (keep only verification + minimal solving)
  - Correct-only (keep only self-correction + minimal solving)
  - Minimal (direct solution, no meta-cognitive behaviors)
- **Cost**: ~$150 API for trace generation + annotation
- **GPU**: 0 (API-based)

**Phase 2 (Day 2-3): Factorial Training — 24×H200 Fully Utilized**
- Base models: Qwen3-1.7B, Qwen3-4B, Qwen3-8B (3 scales)
- Training paradigms: SFT (Day 2), then RL/GRPO on top (Day 3)
- 7 data conditions × 3 model sizes × 2 paradigms = 42 training runs
  - With 24×H200: run ~6-8 small jobs in parallel (1.7B: 1 GPU each, 4B: 2 GPUs, 8B: 4 GPUs)
  - Each SFT run: ~2-4 hours on 5K examples
  - Each RL run: ~4-8 hours
- Framework: TRL or OpenRLHF (proven, user has experience)
- **GPU**: 24×H200 fully saturated for 2 days

**Phase 3 (Day 4): Comprehensive Evaluation**
- In-domain: GSM8K, MATH500
- OOD math: AIME2024, AMC2023 (harder versions)
- Cross-domain: HumanEval+, MBPP+ (code), ARC-Challenge (science)
- Robustness: Perturbed versions of GSM8K (number swap, context change)
- Behavioral analysis: Count frequency of each behavior in model outputs (do trained behaviors persist? do they transfer?)
- **GPU**: Inference only, fast

**Phase 4 (Day 5): Analysis + Writing**
- Compute causal effect of each behavior: Δ(accuracy) = acc(with behavior) - acc(without behavior)
- Cross-domain transfer matrix: which behaviors trained on math help code, and vice versa
- Scale analysis: do behavioral effects change with model size?
- SFT vs RL comparison: does RL amplify certain behaviors more than others?
- Write paper in EMNLP format

### Key Figures (Planned)
1. **Behavioral Contribution Heatmap**: 5 behaviors × 6 eval tasks × 3 scales → which behaviors help where
2. **Cross-Domain Transfer Matrix**: Behavior trained on domain X, evaluated on domain Y
3. **Behavioral Amplification by RL**: How RL changes the frequency/effectiveness of each behavior compared to SFT
4. **Scale Effect**: Does model size interact with behavioral sensitivity?
5. **The "Behavioral Diversity Sweet Spot"**: Performance vs. number of behaviors in training data

### Why This Paper Is Publishable Regardless of Results

| Outcome | Paper Contribution |
|---------|-------------------|
| Verification is most transferable | Practical guideline: prioritize verification in training data |
| Self-correction hurts in-domain | Counter-intuitive finding: challenges current practice of maximizing self-correction |
| All behaviors contribute equally | Negative result: behavioral composition doesn't matter → current practice is fine |
| Behaviors interact non-linearly | Complex finding: motivates future work on behavioral data curation |
| Scale changes behavioral sensitivity | Scaling insight: small models need different behavioral mix than large ones |

### Risk Analysis
- **Result risk: VERY LOW** — factorial design guarantees findings regardless of direction
- **Execution risk: MEDIUM** — 42 training runs in 2 days is tight but feasible with 24×H200
- **Reviewer risk: LOW** — clean experimental design, clear contribution, EMNLP-style empirical paper
- **Competition risk: LOW** — no paper does this exact factorial behavioral anatomy

### Novelty Score: 8/10

**Why 8/10**: 
- The QUESTION is novel: nobody has done a clean factorial experiment on individual reasoning behavior contributions
- Stanford paper asked "do behaviors enable RL?" (binary), we ask "which behaviors cause what downstream effect?" (multi-dimensional)
- Connects to multiple hot areas (reasoning, RL, distillation, data curation) without being in any of their crowded spaces
- Result-independent publishability

**Why not 9/10**:
- Not a fundamentally new technique/algorithm — it's an analysis paper
- The behavioral taxonomy (decompose/verify/correct/explore/summarize) is somewhat standard
- Someone could argue "obvious extension of Stanford paper" (but Stanford only did 1 task)

### Working Title Options
1. "The Anatomy of Useful Reasoning: Which Cognitive Behaviors in Training Data Actually Transfer?"
2. "Not All Reasoning is Created Equal: A Factorial Study of Behavioral Components in CoT Training"
3. "Beyond Correct Traces: How Reasoning Behaviors Shape Generalization in Language Models"

---

## Final Recommendation

### 🏆 Direction F: "Reasoning Style Causal Anatomy" — STRONGLY RECOMMENDED

**Core insight**: The same correct answer, embedded in different reasoning behaviors, produces dramatically different downstream models.

**Why it meets ALL constraints**:
- ✅ **GPU-intensive**: 42 training runs, 24×H200 fully utilized for 2 days
- ✅ **Novelty ≥ 7/10**: 8/10 — no factorial behavioral anatomy study exists
- ✅ **5-day feasible**: Clear 5-phase plan, parallelizable training
- ✅ **Non-overlapping with Paper 1**: Paper 1 = failure analysis of code agents, Paper 2 = behavioral training data science
- ✅ **Insight-driven, not engineering**: Core question is "what matters?" not "how to build X"
- ✅ **Result-independent**: Publishable regardless of which direction results go
- ✅ **EMNLP-appropriate**: Empirical analysis with actionable findings
- ✅ **Connects to user's expertise**: Code + reasoning + training, builds on behavioral scaffolding knowledge from submission1

### Backup: Direction D (if Direction F seems too ambitious)
Direction D is a more focused version: extend Stanford's Cognitive Behaviors paper to multiple domains and scales. Lower novelty (7.5/10) but also lower execution risk.

### NOT Recommended
- A (Thinking Token Attribution): 6/10 novelty, 5+ competitors
- B (RL vs SFT Representation): 5/10 novelty, NeurIPS'25 best paper already here
- C (Agent Planning via RL): 6/10 novelty, high execution risk
- E (Reward Hacking Code Agents): 4/10 novelty, ICML'26 paper already comprehensive

---

## Appendix: Key References

### Direction F (Primary Recommendation) — Related Work
1. Gandhi et al. (2025). "Cognitive Behaviors that Enable Self-Improving Reasoners." arXiv:2503.01307. — 4 behaviors on Countdown game, 3B models only
2. Ding et al. (2025). "Do Thinking Tokens Help or Trap?" arXiv:2506.23840. — Thinking trap phenomenon, DuP-PO solution
3. Matsutani et al. (2025). "RL Squeezes, SFT Expands." arXiv:2509.21128, NeurIPS 2025. — Trajectory topology of RL vs SFT
4. Yue et al. (2025). "Does RL Really Incentivize Reasoning?" arXiv:2504.13837, NeurIPS 2025 Best Paper. — RL doesn't create new patterns
5. Yuan et al. (2025). "Not All Tokens Are What You Need In Thinking." arXiv:2505.17827. — CTS compression framework
6. Chen et al. (2025). "Unveiling Key Factors for CoT Distillation." ACL 2025. — Granularity, format, teacher effects
7. Pang et al. (2025). "Reasoning Pattern Matters." arXiv:2510.12643. — Pattern > rationale quality
8. Dixit et al. (2026). "Project Aletheia: Verifier-Guided Distillation of Backtracking." arXiv:2601.14290. — Backtracking distillation for 7B
9. Cheng et al. (2025). "Guru: Revisiting RL for Reasoning Cross-Domain." arXiv:2506.14965. — 92K examples, 6 domains, transfer patterns
10. DC-CoT (2025). arXiv:2505.18759. — Data-centric CoT distillation benchmark

### Competition Table
| Paper | Behaviors Studied | Tasks | Scales | Paradigm | Factorial? |
|-------|------------------|-------|--------|----------|------------|
| Stanford (2503.01307) | 4 (verify, backtrack, subgoal, backward chain) | 1 (Countdown) | 1 (3B) | RL only | No |
| Ours (proposed) | 5 (decompose, verify, correct, explore, summarize) | 6+ (math, code, science, OOD) | 3 (1.7B, 4B, 8B) | SFT + RL | **Yes** |
| ACL'25 Key Factors | N/A (format, granularity) | 7 (math, commonsense) | 7 models | SFT only | Partial |
| DC-CoT | N/A (data manipulation) | math | 1 | SFT only | No |
