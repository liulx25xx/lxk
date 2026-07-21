# Paper Outline: Beyond Text Matching

**Working Title**: "Beyond Text Matching: Failure Taxonomy, Adaptive Scaffolding, and Recovery Strategies for Code Agents"

**Venue**: EMNLP 2026 (ARR May cycle, deadline 2026-05-25 AoE)
**Format**: ACL Long Paper, 8 pages + references (camera-ready 9 pages)
**Template**: `paper1/template/` (ACL style)

---

## Story Arc (Introduction Logic)

```
1. CODE AGENTS ARE POWERFUL BUT FRAGILE
   → SWE-bench Verified: 79% resolve rate, but failures are poorly understood

2. PRIOR WORK DISCOVERED BEHAVIORAL SCAFFOLDING (submission1)
   → Feedback effectiveness depends on behavioral strategy, not diagnostic accuracy
   → 56pp reversal on str_replace mismatch (re-read 62% vs scan 6%)

3. BUT: Only 1 failure type, 50 samples, no automation
   → str_replace text mismatch ≠ the full failure landscape
   → No one knows if behavioral scaffolding generalizes across failure types
   → No automated system to detect failure type and select strategy

4. WE PROVIDE THREE EXTENSIONS
   → (a) 5-type failure taxonomy + cross-type scaffolding validation
   → (b) Error cascade analysis + Checkpoint-and-Backtrack (C&B) method
   → (c) Automated Failure Classifier + Strategy Selector pipeline

5. KEY FINDINGS
   → Behavioral scaffolding generalizes, but optimal strategy differs per failure type
   → 40-60% of agent steps are wasted post-first-error (cascade waste)
   → C&B improves resolve rate by X% while reducing token cost by Y%
   → Automated pipeline achieves Z% of oracle-strategy performance

6. IMPLICATIONS
   → Agent retry loops should be failure-type-aware, not one-size-fits-all
   → Cascade-breaking is more cost-effective than unlimited retries
   → Simple classifiers can operationalize behavioral scaffolding principles
```

---

## Section-by-Section Plan

### 1. Introduction (~1.0 page, ~800 words)

**Content**:
- Open with the code agent paradox: high resolve rates but fragile failure modes
- Introduce behavioral scaffolding (submission1): correct diagnosis ≠ effective recovery
- State the three gaps: narrow scope (1 type), no cascade analysis, no automation
- Present our three-part contribution clearly
- Close with key findings summary

**Key Claims**:
- C1: Behavioral scaffolding is a general principle that holds across 5+ failure types, but the optimal recovery strategy is failure-type-dependent
- C2: Error cascades waste 40-60% of agent computation; early detection + backtrack recovers X% of these cases
- C3: A simple automated pipeline (classifier + selector) can operationalize behavioral scaffolding with Z% of oracle performance

**Experiment Support**: All three parts feed into these claims

**Citation Notes**: submission1 (anonymous cite), SWE-bench, AgentDebug, Reflexion, Self-Refine

---

### 2. Related Work (~0.7 page, ~550 words)

**Structure**: 3 paragraphs, each ending with gap → our boundary

**Paragraph 1: Agent Feedback and Self-Correction**
- Field sentence: "Several approaches have explored how LLMs can recover from errors during autonomous task execution."
- Works: Reflexion (Shinn et al., 2023), Self-Refine (Madaan et al., 2023), Self-Debugging (Chen et al., 2024; Jiang et al., 2025 ACL), submission1 (anonymous)
- What they solve: self-correction loops, feedback strategies, behavioral scaffolding principle
- Gap: "All prior work studies single failure modes or aggregate success rates; no work systematically maps how different failure types respond to different recovery strategies."
- Our boundary: "We extend behavioral scaffolding from one failure type to five, revealing type-dependent optimal strategies."

**Paragraph 2: Code Agent Failure Analysis**
- Field sentence: "Understanding why code agents fail is prerequisite to designing effective recovery."
- Works: AgentDebug (2025), MASFT (2025), SWE-bench error analyses, Agent-Reliability-Bench (2026)
- What they solve: failure taxonomies, modular debugging
- Gap: "Existing taxonomies classify failures post-hoc but do not study error propagation within trajectories or connect failure types to recovery strategies."
- Our boundary: "We quantify error cascades within trajectories and link failure types to actionable recovery strategies."

**Paragraph 3: Agent Recovery and Backtracking**
- Field sentence: "Recovering from errors during multi-step execution remains an open challenge for autonomous agents."
- Works: Tree-of-Thought (Yao et al., 2024), Agent-R (2025), LATS (Zhou et al., 2024), GiGPO (2025)
- What they solve: search-based recovery, self-reflective training, credit assignment
- Gap: "These methods either require training (Agent-R, GiGPO) or treat all errors uniformly (ToT, LATS); none use failure-type-specific backtracking informed by cascade analysis."
- Our boundary: "We propose Checkpoint-and-Backtrack (C&B), a training-free, cascade-aware recovery strategy."

**Citation Notes**: ~15-20 citations in Related Work section

---

### 3. Failure Taxonomy and Cross-Type Scaffolding (~1.8 pages, ~1400 words) — Part 1

**3.1 Task Setup and Data Collection** (~0.4 page)
- SWE-bench Verified subset: 200 instances
- Agent framework: mini-SWE-agent (bash-only, simple scaffold)
- 4 models: GPT-4o-mini (main), GPT-4.1 (high-cap), DeepSeek V4 (cross-family), Claude Sonnet 4 (behavioral contrast)
- Trajectory collection: full action-observation logs per instance per model

**3.2 Failure Type Definitions** (~0.5 page)
- **Type 1 — Localization Failure**: Agent identifies wrong file or wrong function/class
  - Criterion: edited file/function does not contain the bug
  - Prevalence: expected ~25-30% of failures
- **Type 2 — Edit-Application Failure**: Correct location but edit command fails (str_replace mismatch, syntax error in patch)
  - Criterion: edit tool returns error or produces invalid code
  - Prevalence: expected ~15-20% (submission1's focus)
- **Type 3 — Logic Error**: Edit applies successfully but introduces wrong logic
  - Criterion: edit applies, tests fail on logic (not syntax/import)
  - Prevalence: expected ~25-30%
- **Type 4 — Test Misinterpretation**: Agent misunderstands the test requirements
  - Criterion: agent's stated plan contradicts test expectations
  - Prevalence: expected ~10-15%
- **Type 5 — Planning/Strategy Failure**: Agent chooses fundamentally wrong approach
  - Criterion: overall approach is misguided (e.g., modifying wrong module)
  - Prevalence: expected ~10-15%

**3.3 Cross-Type Scaffolding Experiments** (~0.5 page)
- Per failure type: design 2-3 behavioral strategies (see §3.2 strategy table)
- Test each strategy × type × model combination
- Core analysis: does behavioral scaffolding generalize? Is optimal strategy type-dependent?

**3.4 Results and Findings** (~0.4 page)
- Scaffolding effectiveness per type (heatmap)
- Strategy × type interaction effects
- Model capability thresholds per type
- Key finding: scaffolding generalizes but optimal strategy is type-specific

**Key Claims for §3**:
- Behavioral scaffolding principle holds across all 5 types
- Optimal strategy differs significantly per failure type
- Capability threshold varies by failure type (some types need stronger models)

**Experiment Support**: EXP-003 (trajectory collection), EXP-004 (annotation), EXP-005 (scaffolding tests)

---

### 4. Error Cascade Analysis (~1.3 pages, ~1000 words) — Part 2

**4.1 Cascade Annotation and Metrics** (~0.4 page)
- Annotate each failed trajectory: first-error step, cascade length, wasted steps, recovery attempts
- Metrics: Cascade Length (CL), Waste Ratio (WR = wasted_steps / total_steps), Recovery Success Rate (RSR)
- Annotation method: LLM-assisted (GPT-4o-mini) + rule-based heuristics + human spot-check

**4.2 Cascade Analysis Results** (~0.4 page)
- Distribution of cascade lengths by failure type
- Waste ratio statistics (expected: 40-60% of steps wasted)
- Localization failures → longest cascades; edit failures → shortest
- Recovery attempt frequency and success rates
- Cross-model comparison: do stronger models cascade less?

**4.3 Checkpoint-and-Backtrack (C&B)** (~0.5 page)
- Algorithm description (3 components: checkpoint placement, error detection, backtrack execution)
- 3 variants: C&B-Oracle, C&B-Heuristic, C&B-LLM
- Comparison against: Baseline (no recovery), Naive Retry (restart from scratch)
- Results: resolve rate improvement, token cost reduction, per-type effectiveness

**Key Claims for §4**:
- First quantification of error cascades in code agent trajectories
- Localization errors produce the longest cascades (highest waste)
- C&B-Heuristic improves resolve rate by X% with Y% fewer tokens than naive retry

**Experiment Support**: EXP-006 (cascade annotation), EXP-007 (cascade analysis), EXP-008/009/010 (C&B experiments)

---

### 5. Automated Adaptive Scaffolding (~1.5 pages, ~1200 words) — Part 3

**5.1 System Architecture** (~0.3 page)
- Pipeline: Agent trajectory → Failure Classifier → Strategy Selector → Recovery Action
- Two-stage: (1) detect failure type from trajectory prefix, (2) select and apply optimal strategy
- Diagram of the full pipeline

**5.2 Failure Classifier** (~0.4 page)
- Input: trajectory (action-observation sequence) up to failure point
- Output: one of 5 failure types + confidence score
- Implementation options:
  - Option A: LLM-as-classifier (GPT-4o-mini, few-shot)
  - Option B: Fine-tuned Qwen3-4B on trajectory data (use H200 GPUs)
- Evaluation: classification accuracy, F1 per type

**5.3 Strategy Selector** (~0.3 page)
- Input: classified failure type
- Output: behavioral strategy (from §3's type-strategy mapping)
- Implementation: lookup table (from §3 results) + confidence-weighted fallback
- Integration with C&B: when to scaffold vs when to backtrack

**5.4 End-to-End Evaluation** (~0.5 page)
- Full pipeline on held-out test set (50 instances)
- Comparison: No recovery / Random strategy / Oracle strategy / Our pipeline
- Metrics: resolve rate, token cost, recovery success rate
- Ablation: classifier-only, selector-only, full pipeline
- Key result: automated pipeline achieves Z% of oracle performance

**Key Claims for §5**:
- Simple LLM-based classifier achieves high accuracy (>80%) on failure type detection
- Automated pipeline recovers significant portion of failures without human intervention
- Pipeline is model-agnostic and framework-agnostic

**Experiment Support**: EXP-011 (classifier), EXP-012 (selector), EXP-013 (end-to-end), EXP-014 (ablation)

---

### 6. Discussion and Analysis (~0.7 page, ~550 words)

**Content**:
- When to scaffold vs when to backtrack: decision framework
- Why behavioral scaffolding works across types (strategy-steering hypothesis)
- Cost-effectiveness analysis: scaffolding + C&B vs unlimited retries
- Connection to submission1: our results confirm and extend the behavioral scaffolding principle
- Implications for agent framework design

---

### 7. Limitations (~0.3 page, ~250 words)

**Content**:
- SWE-bench Verified only (may not generalize to other agent tasks)
- Closed-source models as primary subjects (replication concerns)
- 5 failure types may not be exhaustive
- C&B uses fixed checkpoint placement (adaptive placement is future work)
- Strategy designs are hand-crafted per type (not learned)

---

### 8. Conclusion (~0.2 page, ~150 words)

**Content**: Crisp restatement of three contributions + one-line future direction

---

## Page Budget

| Section | Pages | Words |
|---------|-------|-------|
| Introduction | 1.0 | ~800 |
| Related Work | 0.7 | ~550 |
| §3 Failure Taxonomy + Scaffolding | 1.8 | ~1400 |
| §4 Error Cascade + C&B | 1.3 | ~1000 |
| §5 Automated Pipeline | 1.5 | ~1200 |
| Discussion | 0.7 | ~550 |
| Limitations | 0.3 | ~250 |
| Conclusion | 0.2 | ~150 |
| **Total** | **7.5** | **~5900** |
| Figures + Tables overhead | 0.5 | — |
| **Grand Total** | **8.0** | — |

---

## Figures and Tables Plan

### Figures

| ID | Content | Section | Size | Purpose |
|----|---------|---------|------|---------|
| **Figure 1** | System overview diagram: 3-part contribution (taxonomy → cascade → pipeline) | §1 Intro | 1-column | Roadmap for the paper |
| **Figure 2** | Failure type distribution across 4 models (stacked bar chart) | §3.2 | 1-column | Show failure type prevalence |
| **Figure 3** | Scaffolding effectiveness heatmap: strategy × failure type × model | §3.4 | 2-column | Core result of Part 1 — shows type-dependent optimal strategy |
| **Figure 4** | Cascade length distribution by failure type (violin plot or box plot) | §4.2 | 1-column | Visualize cascade severity by type |
| **Figure 5** | Resolve rate comparison: Baseline vs Retry vs C&B variants (grouped bar) | §4.3 | 1-column | Core result of Part 2 |
| **Figure 6** | End-to-end pipeline diagram (classifier → selector → action) | §5.1 | 1-column | Architecture of Part 3 |

### Tables

| ID | Content | Section | Purpose |
|----|---------|---------|---------|
| **Table 1** | Failure type definitions with criteria and examples | §3.2 | Reference for the 5 types |
| **Table 2** | Behavioral strategies per failure type (type × 2-3 strategies) | §3.3 | Strategy design |
| **Table 3** | Cross-type scaffolding results: recovery rate per (type, strategy, model) | §3.4 | Quantitative results of Part 1 |
| **Table 4** | Cascade statistics: mean CL, WR, RSR per failure type per model | §4.2 | Quantitative results of Part 2 |
| **Table 5** | C&B comparison: resolve rate, token cost, per-type breakdown | §4.3 | Method comparison |
| **Table 6** | Failure classifier accuracy: per-type precision/recall/F1 | §5.2 | Classifier evaluation |
| **Table 7** | End-to-end pipeline results: No-recovery / Random / Oracle / Ours | §5.4 | Core result of Part 3 |

### Appendix Tables/Figures (no page limit)

| ID | Content | Purpose |
|----|---------|---------|
| Table A1 | Full per-model, per-type scaffolding results | Complete results |
| Table A2 | Prompt templates for each failure type × strategy | Reproducibility |
| Table A3 | Failure annotation guidelines + inter-annotator agreement | Methodology |
| Table A4 | Classifier confusion matrix | Detailed classifier analysis |
| Table A5 | Statistical significance tests (bootstrap CIs) | Rigor |
| Figure A1 | Example trajectories for each failure type | Qualitative illustration |
| Figure A2 | C&B execution traces (before/after backtrack) | Method illustration |
| Figure A3 | Classifier confidence distribution per type | Calibration analysis |

---

## Citation Strategy

### Must-Cite (Core References)

**Behavioral Scaffolding Foundation**:
1. **[Anonymous] submission1** — "When Correct Feedback Fails: Behavioral Scaffolding for LLM Code-Edit Repair" (under review; anonymous cite as "Anonymous, 2026")

**SWE-bench and Agent Frameworks**:
2. Jimenez et al. (2024) — SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
3. Yang et al. (2024) — SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
4. Wang et al. (2024/2025) — OpenHands/OpenDevin
5. Xia et al. (2024) — Agentless: Demystifying LLM-based Software Engineering Agents (ICML 2025)

**Self-Correction and Feedback**:
6. Shinn et al. (2023) — Reflexion: Language Agents with Verbal Reinforcement Learning (NeurIPS 2023)
7. Madaan et al. (2023) — Self-Refine: Iterative Refinement with Self-Feedback (NeurIPS 2023)
8. Jiang et al. (2025) — Revisiting the Self-Debugging Capability of LLMs (ACL 2025)
9. Olausson et al. (2024) — Is Self-Repair a Silver Bullet for Code Generation? (ICLR 2024)

**Failure Analysis**:
10. AgentDebug (2025) — Modular failure classification + debugging framework
11. MASFT (2025) — Multi-Agent System Failure Taxonomy

**Agent RL (Context)**:
12. DeepSWE (2025) — Pure RL for SWE-bench
13. GiGPO (NeurIPS 2025) — Two-level grouped advantage for agent RL
14. Kimi-Dev (2025) — Agentless as skill prior

**Recovery and Search**:
15. Yao et al. (2024) — Tree of Thoughts (NeurIPS 2024)
16. Zhou et al. (2024) — LATS: Language Agent Tree Search

**LLM Foundations (context only)**:
17. GPT-4 technical report (OpenAI, 2023)
18. DeepSeek V4 technical report
19. Claude model card (Anthropic)

### Nice-to-Cite (If Space Allows)
- Agent-R (2025) — self-reflective agent training
- SWE-smith (2025) — synthetic SWE training data
- mini-coder (2026) — small model agents
- Agent-Omit (ICML 2026) — agent efficiency
- SAUP (ACL 2025) — uncertainty propagation in agents
- R2E-Gym (2025) — RL environment for SWE
- CodeAct (2024) — code as agent actions

**Target**: ~30-40 citations total (natural placement, not padding)

---

## Relationship to Submission1

- **Anonymous citation**: "Anonymous (2026) established behavioral scaffolding for edit-application repair; we generalize to the full spectrum of agent failures"
- **Non-overlapping**: submission1 = 1 type × 12 models × mechanism insight; we = 5 types × 4 models × cascade + automation
- **Complementary**: submission1 discovers the principle, we operationalize it
- **Self-citation handling**: if submission1 is under review at NeurIPS, cite as anonymous. If accepted, can cite normally. ARR allows citing concurrent submissions with appropriate anonymization.

---

## Writing Timeline

| Day | Writing Task |
|-----|-------------|
| Day 1-3 | — (experiments running) |
| Day 4 AM | Draft §3 (Failure Taxonomy) — results should be ready |
| Day 4 PM | Draft §4 (Error Cascade + C&B) |
| Day 5 AM | Draft §5 (Automated Pipeline) + §1 (Intro) + §2 (Related Work) |
| Day 5 PM | §6-8 (Discussion/Limitations/Conclusion) + Abstract + polish |
| Day 5 EVE | Final compile, proofread, submit |
