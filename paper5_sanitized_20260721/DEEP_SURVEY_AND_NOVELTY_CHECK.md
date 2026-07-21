# Deep Survey & Novelty Check: SelfCurriculum
## Self-Evolving Curriculum Learning for Domain-Specific LLM Reasoning

**Date**: 2026-05-16  
**Purpose**: Brutally honest novelty assessment for EMNLP 2026 submission  
**Verdict**: See Section 5

---

## 1. Core Competitor Deep Analysis

### 1.1 R-Zero (arXiv:2508.05004) — MOST DIRECT COMPETITOR

**Authors**: Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, Dong Yu (Tencent AI Lab)

**Methodology**:
- **Challenger-Solver co-evolution**: Starting from a single base LLM, instantiates two independent models: a Challenger (generates problems at the edge of Solver capability) and a Solver (solves increasingly challenging tasks)
- **Reward design**: Challenger is rewarded for proposing tasks the Solver *almost* can solve (near the boundary); Solver is rewarded for solving tasks
- **No external data**: Fully autonomous, no pre-existing tasks or labels
- **Optimization**: Both models optimized separately via RL (GRPO-based)

**Key Results**:
- Qwen3-4B-Base: +6.49 on math-reasoning, +7.54 on general-domain reasoning
- Latest version (v4, Feb 2026) extends to general-domain

**Domains Tested**: Math-reasoning + general-domain reasoning (details unclear from abstract — likely still math-centric)

**Limitations We Exploit**:
1. **No explicit domain conditioning** — generates problems without domain-specific structure
2. **No per-domain curriculum control** — treats all difficulty uniformly
3. **Verification method unclear** — likely uses code execution or rule-based for math; no explicit multi-domain verifier
4. **Single difficulty dimension** — no composite difficulty across domains

**CRITICAL RISK**: R-Zero v4 (Feb 2026) claims "+7.54 on general-domain reasoning" — need to verify if this is actually multi-domain or just non-math logic tasks.

---

### 1.2 SQLM — Self-Questioning Language Models (arXiv:2508.03682)

**Authors**: Lili Chen, Mihir Prabhudesai, Katerina Fragkiadaki, Hao Liu, Deepak Pathak (CMU)

**Methodology**:
- **Asymmetric self-play**: Proposer generates questions given topic prompt, Solver attempts to answer
- **Verification**: 
  - Math/arithmetic: Majority voting over N solver outputs as proxy for correctness
  - Code: Proposer generates unit tests for verification
- **Proposer reward**: Gets reward if problem is not too easy (all agree) or too hard (none agree) — reward of 1 when 1 < majority_count < N
- **Solver reward**: Binary — matches majority answer or not

**Key Results**:
- Three-digit multiplication: 0.791 → 0.948
- Linear equations: 0.440 → 0.600
- Codeforces: 0.320 → 0.391

**Domains**: Arithmetic, algebra, coding (ALL verifiable domains)

**Limitations We Exploit**:
1. **Only verifiable domains** (math, code) — no science, law, medicine
2. **No difficulty control beyond implicit** — only "not too easy, not too hard" binary signal
3. **No domain mixing** — each task trained independently
4. **No cross-model verification** — single model majority voting only
5. **Small scale** — Qwen2.5-3B-Instruct only

---

### 1.3 TTRL — Test-Time Reinforcement Learning (arXiv:2504.16084)

**Authors**: Yuxin Zuo, Kaiyan Zhang, Li Sheng, et al.

**Methodology**:
- Uses **majority voting** over multiple model outputs on *unlabeled test data* as reward signal
- Performs RL training at test time — model self-evolves using consensus as proxy truth
- No ground-truth labels needed

**Key Results**:
- Qwen-2.5-Math-7B: ~211% improvement on AIME 2024 (pass@1)
- Surpasses initial model's maj@N upper bound

**Domains**: Primarily math reasoning (AIME). Claims broader applicability but demonstrated mainly on math.

**Limitations We Exploit**:
1. **Test-time only** — not a training-time curriculum framework
2. **Single-model voting** — no cross-model agreement signal
3. **No problem generation** — works on fixed existing test data
4. **No domain adaptation** — majority voting reliability differs across domains (not studied)

---

### 1.4 Crossing the Reward Bridge (arXiv:2503.23829)

**Authors**: Yi Su, Dian Yu, Linfeng Song, et al. (Tencent AI Lab)

**Methodology**:
- **Trained generative verifier (7B LLM)**: Instead of rule-based verification, trains a generative scoring model that produces soft reward signals
- **Key insight**: "Binary verification judgments on broad-domain tasks exhibit high consistency across various LLMs provided expert-written reference answers exist"
- **Cross-domain generative reward model**: Small 7B LLMs trained without extensive domain-specific annotation

**Domains**: Medicine, chemistry, psychology, economics, education

**Key Results**: Significantly outperforms Qwen2.5-72B and DeepSeek-R1-Distill-Qwen-32B on free-form settings

**Limitations We Exploit**:
1. **REQUIRES TRAINING A VERIFIER** — not training-free; needs compute and data for 7B verifier
2. **Relies on expert-written reference answers** — not self-evolving; needs curated reference data
3. **No problem generation** — uses existing datasets
4. **No curriculum** — no difficulty progression
5. **Our CPV is training-free** — this is a key differentiator

---

### 1.5 RLPR — Extrapolating RLVR without Verifiers (arXiv:2506.18254)

**Authors**: Tianyu Yu, Bo Ji, et al. (Tsinghua + NUS)

**Methodology**:
- Uses LLM's **own token probability** of generating correct reference answer as reward signal
- Addresses high variance through "prob-to-reward" and stabilizing methods
- Verifier-free: no external verification model needed

**Domains**: 4 general-domain benchmarks + 3 math benchmarks (TheoremQA, Minerva)

**Key Results**: Outperforms VeriFree by 7.6 on TheoremQA, surpasses General-Reasoner by 1.6 avg

**Limitations We Exploit**:
1. **Still requires reference answers** — not self-evolving; needs ground-truth answers in the dataset
2. **No problem generation** — works on fixed datasets
3. **No curriculum** — no difficulty progression
4. **High variance acknowledged** — the probability reward is inherently noisy
5. **Different verification philosophy** — we combine SC + cross-model rather than probability-based

---

### 1.6 VCRL — Variance-based Curriculum RL (arXiv:2509.19803)

**Authors**: Guochao Jiang, Wenfeng Feng, et al. (Alibaba Cloud)

**Methodology** (FULL PAPER READ):
- **Core insight**: Reward variance across rollouts reflects sample difficulty — high variance = boundary of model capability
- **Variance-based Dynamic Sampling**: Computes normalized variance p = σ²/σ²_max for each query; filters by threshold κ
- **Replay Learning**: Memory bank stores high-variance samples; momentum-based priority update
- **Algorithm**: Sample batch → compute p for each query → filter low-p samples → replace with memory bank samples → RL update → update memory bank

**Key Results**: 
- Qwen3-4B-Base: avg 49.43 vs GRPO 41.76, DAPO 40.60
- Qwen3-8B-Base: avg 57.76 vs GRPO 50.25, GSPO 53.09

**Domains**: **MATH ONLY** (AIME-2024, AIME-2025, MATH500, OlympiadBench, AMC23)

**Limitations We Exploit**:
1. **Math only** — no multi-domain
2. **No problem generation** — uses fixed dataset (DAPO-Math-17K)
3. **No domain-specific adaptation** — single curriculum for single domain
4. **Variance proxy is coarse** — works for binary rewards but doesn't distinguish domain-specific difficulty
5. **No verifier innovation** — assumes standard rule-based verifier

---

### 1.7 CDAS — Competence-Difficulty Alignment Sampling (arXiv:2505.17652)

**Authors**: Deyang Kong, Qi Guo, et al. (Peking University + Meituan)

**Methodology** (FULL PAPER READ):
- **Stable difficulty estimation**: Aggregates historical performance discrepancies across problems (not single-estimate)
- **Fixed-point competence quantification**: Models current model competence as a scalar
- **Adaptive sampling**: Selects problems whose difficulty aligns with model's current competence using fixed-point system

**Key Results**: 
- Highest average accuracy (45.89%) on math benchmarks
- 2.33x faster than Dynamic Sampling (DAPO variant)

**Domains**: **MATH ONLY** (mathematical RL training)

**Limitations We Exploit**:
1. **Math only** — no domain extension
2. **No problem generation** — selects from existing pool
3. **Single domain** — no multi-domain mixing or domain-conditioned curriculum
4. **No verifier contribution** — focuses purely on sampling strategy

---

### 1.8 OpenSIR — Open-Ended Self-Improving Reasoner (arXiv:2511.00602)

**Authors**: Wai-Chung Kwan, et al.

**Methodology**:
- Teacher-student self-play: LLM alternates roles to generate and solve novel problems
- **Dual optimization**: Rewards problems that challenge appropriately + explore distinct concepts (difficulty + diversity)
- **Open-ended learning**: Starts from single trivial seed problem, progresses to advanced math
- No external supervision or verifiers

**Key Results**:
- Llama-3.2-3B: GSM8K 73.9→78.3, College Math 28.8→34.4
- Gemma-2-2B: GSM8K 38.5→58.7

**Domains**: **MATH ONLY** (GSM8K, College Math)

**Limitations We Exploit**:
1. **Math only** — explicitly "from basic to advanced mathematics"
2. **No domain conditioning** — no mechanism for science/law/medicine
3. **No explicit verifier** — relies on implicit correctness assessment
4. **Small gains on harder benchmarks** — College Math only +5.6

---

## 2. CRITICAL NEWLY-DISCOVERED COMPETITORS

### 2.1 ⚠️ SEC — Self-Evolving Curriculum for LLM Reasoning (arXiv:2505.14970) — MAJOR THREAT

**Authors**: Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, **Yoshua Bengio**, Ehsan Kamalloo

**Methodology**:
- Formulates curriculum selection as **non-stationary Multi-Armed Bandit (MAB)** problem
- Each problem category (difficulty level or problem type) = one arm
- Uses **absolute advantage from policy gradient** as proxy for immediate learning gain
- Updates curriculum policy using **TD(0)** method
- Operates concurrently with RL fine-tuning

**Domains**: **THREE domains** — planning, inductive reasoning, AND mathematics

**Key Claims**: 
- "Better skill balance when fine-tuning simultaneously on multiple reasoning domains"
- "Better generalization to harder, out-of-distribution test problems"

**NOVELTY THREAT ASSESSMENT**: 🔴 **HIGH**
- SEC IS doing multi-domain curriculum RL
- SEC IS doing adaptive difficulty control
- SEC IS doing domain mixing with balance optimization
- BUT: SEC does NOT generate problems (selects from existing pools)
- BUT: SEC has no verifier innovation
- BUT: SEC's domains (planning, inductive reasoning) are still somewhat verifiable
- BUT: SEC does not address the verification challenge for non-verifiable domains

**Our differentiation from SEC**:
1. We GENERATE domain-specific problems (Challenger) — SEC only selects
2. We solve the VERIFICATION problem with CPV — SEC assumes verifiable rewards exist
3. We handle genuinely unverifiable domains (law, medicine) — SEC works on planning/logic
4. We use per-domain difficulty tracking — SEC uses category-level MAB

---

### 2.2 ⚠️ WIST — Web-Grounded Iterative Self-Play Tree (arXiv:2603.22352) — CRITICAL THREAT

**Authors**: Fangyuan Li, Pengfei Li, et al. (March 2026)

**Methodology**:
- **Challenger-Solver self-play with verifiable rewards** on open web data
- **Domain tree expansion**: Incrementally builds domain exploration structure
- **Web corpus retrieval**: Retrieves and cleans path-consistent web data for training environment
- **Adaptive curriculum**: Feeds learnability signals back to update node posteriors

**Domains**: MEDICINE (+14.79 on Qwen3-8B-Base) and PHYSICS (+5.28 on PhyBench)

**Key Results**: +9.8 (Qwen3-4B-Base), +9.7 (OctoThinker-8B) overall

**NOVELTY THREAT ASSESSMENT**: 🔴🔴 **VERY HIGH — THIS IS OUR CLOSEST COMPETITOR**
- WIST uses Challenger-Solver self-play ✓
- WIST is domain-targeted (including medicine!) ✓
- WIST has adaptive curriculum ✓
- WIST generates problems from web data ✓

**Critical differences (our survival arguments)**:
1. **Verification**: WIST uses "verifiable rewards" from web — needs answer extraction from web corpus. We use training-free CPV (SC + cross-model) — no web dependency
2. **Problem generation**: WIST constructs problems from retrieved web documents. We use a domain-conditioned Challenger with explicit difficulty control
3. **Curriculum mechanism**: WIST uses tree-based exploration with posteriors. We use per-domain adaptive controller with difficulty tracking
4. **Self-contained vs web-dependent**: Our approach is fully self-contained; WIST requires web access

**RISK**: WIST is from March 2026. If published before us, it significantly weakens our novelty claim on "extending self-play to non-math domains."

---

### 2.3 ⚠️ RLCCF — RL from Coevolutionary Collective Feedback (arXiv:2508.12338) — MODERATE THREAT

**Authors**: Wenzhen Yuan, Shengji Tang, et al.

**Methodology**:
- **Collective Consistency (CC)**: Diverse ensemble of LLMs voting on outputs provides reward signal
- **Self-Consistency (SC) weighting**: Each model's vote weighted by its own self-consistency score
- **Multi-model coevolution**: Multiple LLMs trained jointly, collectively improving

**NOVELTY THREAT TO CPV**: 🟡 **MODERATE**
- RLCCF COMBINES self-consistency + cross-model agreement ← **This is similar to our CPV!**
- Difference: RLCCF trains multiple separate models jointly (expensive). Our CPV uses a fixed external model as cross-verifier (cheaper, training-free)
- Difference: RLCCF is for math only. We study reliability across domains
- Difference: Our CPV is a verification mechanism; RLCCF is a full training framework

---

### 2.4 Reasoning Curriculum (arXiv:2510.26143)

**Authors**: Bo Pang, et al. (Salesforce)

**Methodology**:
- Two-stage curriculum: Stage 1 = math-only RL (elicit reasoning), Stage 2 = joint RL on mixed-domain data
- Backbone-agnostic, no specialized reward models
- Math-first reasoning then transfers to other domains

**Domains**: Multi-domain suite (unspecified), tested on Qwen3-4B and Llama-3.1-8B

**Threat Level**: 🟡 MODERATE — shows math→general transfer works, but no problem generation or novel verification

---

### 2.5 EDCO — Dynamic Curriculum Orchestration (arXiv:2601.03725)

**Authors**: Jing-Cheng Pang, et al.

**Methodology**: Entropy-based dynamic curriculum for domain-specific LLM fine-tuning

**Domains**: Communication, Medicine, Law (under both SFT and RL settings!)

**Threat Level**: 🟡 MODERATE — Tests on our exact domains! But:
- No problem generation
- No self-play
- No novel verifier
- Static curriculum selection, not self-evolving

---

### 2.6 Other Notable Papers

| Paper | Threat | Key Difference |
|-------|--------|---------------|
| **Absolute Zero** (2505.03335) | LOW | Math+code only, uses code executor as verifier |
| **VeriFree** (2505.21493) | LOW | Probability-based reward, no generation, no curriculum |
| **General-Reasoner** (2505.14652) | LOW | Trains generative verifier (like CRB), no self-play |
| **Med-RLVR** (2502.19655) | LOW | MCQA verification only, no generation, no curriculum |
| **VHG** (2605.06660) | LOW | Math only, three-party self-play with verifier for problem generation |
| **SPARK** (2605.05546) | LOW | KG-based self-play for scientific lit, different focus |
| **DoGe** (2512.06835) | MODERATE | Evolving curriculum for VLMs in specialized domains |
| **SUPERNOVA** (2604.08477) | LOW | Data curation for general RLVR, no self-play |

---

## 3. Novelty Verdict Per Contribution

### Contribution 1: Composite Pseudo-Verifier (CPV) — Self-Consistency + Cross-Model Agreement

**Question**: Has ANYONE combined self-consistency + cross-model agreement as a training-free pseudo-verifier?

**Evidence Found**:
- **RLCCF** (2508.12338): Combines CC (cross-model voting) + SC (self-consistency weighting). 🔴 **PARTIALLY OVERLAPS**. But: trains multiple models jointly, not training-free, math-only.
- **Xue et al. (2502.15845)**: "Verify when Uncertain" — explicitly proposes self-consistency THEN cross-model consistency for hallucination detection. Uses two-stage algorithm switching between SC and cross-consistency. 🔴 **SIGNIFICANT OVERLAP** on the combination idea, but for detection, not RL reward.
- **Tan et al. (2505.17656)**: "Too Consistent to Detect" — shows self-consistent errors differ across LLMs, proposes cross-model probe. Motivates our approach but is for error detection.
- **TTRL**: Majority voting only (single model)
- **SQLM**: Majority voting only (single model)

**Novelty Verdict**: 🟡 **PARTIAL NOVELTY**
- The *combination* of SC + cross-model for pseudo-verification exists conceptually in hallucination detection literature
- The *application as RL reward signal* for training is more novel
- The *training-free* aspect differentiates from RLCCF
- The *domain reliability study* across science/law/medicine is novel
- **MUST CITE**: Xue et al. (2502.15845), RLCCF (2508.12338), Tan et al. (2505.17656)

---

### Contribution 2: Domain-Conditioned Challenger with Difficulty Control

**Question**: Has ANYONE done domain-conditioned problem generation with difficulty control outside math?

**Evidence Found**:
- **WIST** (2603.22352): Generates domain-targeted problems including medicine/physics via web retrieval + Challenger-Solver. 🔴 **OVERLAPS on domain-targeted problem generation**
- **R-Zero**: Problem generation at difficulty boundary, but math-only and not domain-conditioned
- **SQLM**: Topic-conditioned proposer (e.g., "algebra word problems"), but limited to math/code
- **KNIGHT** (2602.20135): KG-driven MCQ generation with difficulty control across domains — but for evaluation, not RL training
- **BD-FDG** (2603.09231): Cognitively layered question generation with difficulty gradient — but for SFT data, not self-evolving RL
- **JudgeAgent** (2509.02097): Dynamic evaluation with difficulty-adaptive generation — evaluation only

**Novelty Verdict**: 🟡 **PARTIAL NOVELTY**
- WIST already does domain-targeted problem generation for RL, weakening this claim
- Our explicit difficulty control mechanism (quantified difficulty labels, multi-level generation) may differ from WIST's web-grounded approach
- Problem generation with difficulty control for non-math domains exists in education/evaluation literature but not as self-evolving RL curriculum
- **Our differentiation**: Explicit difficulty control parameters, domain-conditioned prompting without web dependency, more domains (law, science, medicine simultaneously)

---

### Contribution 3: Adaptive Curriculum Controller (Per-Domain Difficulty Tracking + Domain Mixing)

**Question**: Has ANYONE done per-domain adaptive curriculum with domain mixing for self-evolving RL?

**Evidence Found**:
- **SEC** (2505.14970): MAB-based curriculum across planning, inductive reasoning, math — with "better skill balance when fine-tuning simultaneously on multiple reasoning domains". 🔴 **DIRECTLY OVERLAPS**
- **VCRL** (2509.19803): Variance-based adaptive curriculum — but math-only, single domain
- **CDAS** (2505.17652): Competence-difficulty alignment — but math-only
- **EDCO** (2601.03725): Dynamic curriculum across medicine, law, communication — but not self-evolving, no problem generation
- **Reasoning Curriculum** (2510.26143): Math→multi-domain transfer, but fixed two-stage, not adaptive

**Novelty Verdict**: 🟡 **PARTIAL NOVELTY**
- SEC already does multi-domain curriculum with adaptive selection
- Our per-domain difficulty TRACKING (maintaining domain-specific difficulty curves) adds beyond SEC's MAB formulation
- Our integration with self-generated problems + CPV verification is the combination novelty
- **MUST CITE**: SEC (2505.14970), VCRL (2509.19803), CDAS (2505.17652), EDCO (2601.03725)

---

## 4. Overall Novelty Assessment

### Is the COMBINATION Novel?

**The complete pipeline**: Domain-Conditioned Challenger → CPV Verification → Adaptive Multi-Domain Curriculum Controller

**Closest existing systems**:
| System | Problem Gen | Non-Math Domains | Training-Free Verifier | Multi-Domain Curriculum |
|--------|------------|-----------------|----------------------|----------------------|
| **R-Zero** | ✓ | ✗ (math) | ✗ | ✗ |
| **SQLM** | ✓ | ✗ (math+code) | ✓ (maj voting) | ✗ |
| **WIST** | ✓ | ✓ (med, phys) | ✗ (web-based) | ✓ (tree-based) |
| **SEC** | ✗ | ✓ (plan, logic) | ✗ (assumes) | ✓ (MAB) |
| **RLCCF** | ✗ | ✗ (math) | ✓ (multi-model) | ✗ |
| **TTRL** | ✗ | ✗ (math) | ✓ (maj voting) | ✗ |
| **Ours** | ✓ | ✓ (sci, law, med) | ✓ (CPV) | ✓ (per-domain) |

**Honest Assessment**: 🟡 **THE COMBINATION IS MODERATELY NOVEL, BUT EACH INDIVIDUAL COMPONENT HAS SIGNIFICANT PRIOR ART**

The key risk is **WIST** (March 2026), which already demonstrates:
- Challenger-Solver for non-math domains (medicine, physics)
- Adaptive curriculum for domain-targeted learning
- Self-play problem generation from web

Our remaining novelty over WIST:
1. **Training-free verification** (CPV) vs web-dependent answer extraction
2. **Broader domain coverage** (law!) and explicit multi-domain mixing
3. **Self-contained** (no web access needed at training time)
4. **Explicit difficulty quantification** vs implicit learnability signals

---

## 5. BRUTALLY HONEST SUMMARY

### Strengths of Our Position:
1. **The full combination** (generate + verify + curriculum across 3+ domains) is not exactly replicated by any single paper
2. **Training-free CPV** is genuinely different from Crossing the Reward Bridge's trained verifier
3. **Law domain** is genuinely understudied — no paper does self-evolving RL for legal reasoning
4. **Self-contained** approach (no web dependency) is practically valuable vs WIST

### Weaknesses / Risks:
1. **WIST (2603.22352) is devastatingly close** — same paradigm (Challenger-Solver + domain targeting + curriculum) applied to medicine
2. **SEC (2505.14970) already does multi-domain curriculum RL** with domain mixing
3. **CPV's SC + cross-model combination** exists in hallucination detection (Xue et al., 2502.15845)
4. **Individual components are not novel** — novelty comes ONLY from the specific combination
5. **The "contribution list" reads incremental** to a reviewer who knows WIST + SEC

### Recommendations:
1. **MUST prominently compare against WIST** — position as "self-contained alternative to web-grounded approaches"
2. **MUST cite and differentiate from SEC** — our MAB-inspired idea needs clear positioning
3. **Reframe Contribution 1 (CPV)**: Emphasize the *domain reliability study* — "when does pseudo-verification fail?" This is genuinely unstudied
4. **Emphasize law domain** — least explored territory
5. **Consider adding code/tool-augmented verification for some domains** to differentiate from pure LLM-based approaches
6. **Run head-to-head against WIST** if possible — demonstrate advantages of self-contained approach

---

## 6. Mandatory Citation List

### Must-Cite (directly competing or inspiring):
1. R-Zero (2508.05004) — Challenger-Solver framework ancestor
2. SQLM (2508.03682) — Self-questioning with majority voting
3. TTRL (2504.16084) — Majority voting as RL reward
4. Crossing the Reward Bridge (2503.23829) — Trained verifier for non-math RLVR
5. RLPR (2506.18254) — Verifier-free RLVR for general domains
6. VCRL (2509.19803) — Variance-based curriculum RL
7. CDAS (2505.17652) — Competence-difficulty alignment
8. OpenSIR (2511.00602) — Open-ended self-improving reasoner
9. **SEC (2505.14970)** — Self-Evolving Curriculum for LLM Reasoning ⚠️ CRITICAL
10. **WIST (2603.22352)** — Web-Grounded Self-Play for Domain-Targeted Reasoning ⚠️ CRITICAL
11. **RLCCF (2508.12338)** — Multi-model collective feedback (SC + CC)
12. Absolute Zero (2505.03335) — Zero-data self-play paradigm
13. Med-RLVR (2502.19655) — RLVR in medical domain
14. VeriFree (2505.21493) — Verifier-free general reasoning
15. General-Reasoner (2505.14652) — Generative verifier for broad domains

### Should-Cite (related concepts):
16. Xue et al. (2502.15845) — SC + cross-model consistency for hallucination detection
17. Tan et al. (2505.17656) — Self-consistent errors across LLMs
18. EDCO (2601.03725) — Dynamic curriculum for medicine/law/communication
19. Reasoning Curriculum (2510.26143) — Math→multi-domain transfer
20. DoGe (2512.06835) — Self-evolving curriculum for data-scarce VLM domains
21. SPIRAL (2506.24119) — Self-play games for reasoning
22. VHG (2605.06660) — Verifier-backed hard problem generation
23. DeepSeek-R1 (2501.12948) — Foundation RLVR work
24. DAPO (2503.14476) — RL algorithm baseline
25. GRPO (2402.03300) — Core RL algorithm

---

## 7. Revised Novelty Positioning (Recommended Framing)

Given the competitor landscape, we should NOT frame our paper as:
- ❌ "First to extend self-evolving RL beyond math" (WIST does this)
- ❌ "First to combine SC + cross-model for verification" (RLCCF, Xue et al.)
- ❌ "First multi-domain curriculum for RL" (SEC does this)

We SHOULD frame as:
- ✅ "First **self-contained** (no web/external data) self-evolving framework for **domain-specific** reasoning across science, law, and medicine"
- ✅ "First systematic study of **pseudo-verifier reliability across domains** — when does SC + cross-model agree/disagree, and how does this affect RL training?"
- ✅ "First to integrate **explicit difficulty-controlled generation** with **training-free composite verification** in a unified adaptive curriculum — addressing the challenge that non-math domains lack both problem generators and verifiers simultaneously"
- ✅ "First to demonstrate that the Challenger-Solver paradigm can work WITHOUT web-grounded answers, using only model-internal signals (CPV), across genuinely open-ended domains (law, medicine)"

---

## 8. Risk Matrix

| Risk | Severity | Mitigation |
|------|----------|-----------|
| WIST published and well-cited before EMNLP | HIGH | Run direct comparison; emphasize self-contained advantage |
| SEC published at top venue before EMNLP | MEDIUM | Position as complementary; our MAB formulation differs |
| Reviewer knows RLCCF and sees CPV as incremental | MEDIUM | Emphasize domain reliability analysis as primary CPV contribution |
| R-Zero v4 already covers general domains well | MEDIUM | Verify their "general domain" claim; likely still logic-heavy |
| WIST + SEC combination in reviewer's mind kills us | HIGH | Must show empirical advantage of unified pipeline vs piecemeal |
| Another 2026 paper combines all three before us | HIGH | Submit early; emphasize unique angles (law, self-contained) |

---

## 9. Final Verdict

**Overall Novelty**: 🟡 **MODERATE — Publishable at EMNLP if positioned correctly, but NOT a strong novelty story**

The paper is strongest when framed as:
1. A **systems contribution** — the first complete, self-contained pipeline that works across genuinely diverse domains
2. An **empirical contribution** — systematic study of where pseudo-verification succeeds/fails across domains
3. A **practical contribution** — no web access, no trained verifier, no external data needed

The paper is weakest when individual components are examined in isolation — each has significant prior art. The combination is the novelty, and we must sell it as such.

**Recommendation**: Proceed with submission, but SIGNIFICANTLY revise related work and contribution claims to acknowledge WIST, SEC, and RLCCF. The paper lives or dies on:
1. Strong empirical results showing the unified approach outperforms piecemeal alternatives
2. The domain reliability analysis of CPV (genuinely novel and interesting)
3. Clear demonstration on LAW domain (least-explored territory)
