# 🎯 Cost-Optimized Experiment Execution Plan

**原则**: 先跑已有的公开trajectory, 有效果了再自己跑  
**Deadline**: 2026-05-25 AoE (8 days from today, 5/17)  
**Budget**: 200 RMB/batch (~$28 USD/batch), total ~500 RMB  
**Updated**: 2026-05-17

---

## Architecture Overview: 6-Phase Gated Execution

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5
FREE        FREE        ¥30         ¥60         ¥80         ¥30
Download    Annotate    Validate    Core        Scale-up    Ceiling
Public      by Rules    Small LLM   Scaffolding C&B         + Polish
Trajs                   Batch       Experiments Pipeline

      ↓ GO/NO-GO gate at each transition ↓
```

Each phase has explicit go/no-go criteria. 
**If a phase fails, we STOP spending and pivot** — never 白跑.

---

## ═══════════════════════════════════════
## Phase 0: ZERO-COST Data Foundation (Day 1: 5/17-5/18)
## Cost: ¥0 | Time: 4-6 hours
## ═══════════════════════════════════════

### What to do

**0A. Download public SWE-bench agent trajectories from HuggingFace**

Known public trajectory sources (FREE, no API cost):
```
1. SWE-agent official runs:
   - princeton-nlp/SWE-bench_Verified (official split)
   - SWE-agent leaderboard predictions (GitHub: princeton-nlp/SWE-agent)
   
2. Agentless trajectories:
   - OpenAutoCoder/Agentless (GitHub releases with predictions)
   - Agentless predictions on SWE-bench Verified
   
3. OpenHands/OpenDevin runs:
   - All-Hands-AI/OpenHands (public eval logs)
   - OpenHands leaderboard submissions
   
4. Community leaderboard submissions:
   - HuggingFace datasets tagged "swe-bench"
   - AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results (already used in Paper 6)

5. SWE-bench-docker predictions:
   - Various leaderboard entries with public model_patch files
```

Script to write: `scripts/download_public_trajectories.py`
- Search HuggingFace Hub API for datasets matching "swe-bench" + "trajectory"
- Download Agentless/SWE-agent/OpenHands prediction files from GitHub releases
- Normalize all trajectories to our standard format: {instance_id, steps[], resolved, model}
- Cross-reference with our 200-instance subset

**Target**: Get ≥3 different agents' trajectories on our 200 instances (or significant overlap)

**0B. Build rule-based failure annotator (ZERO API cost)**

We already have `annotate_failures.py` with a `rule_based_classify()` function.
Extend it to work STANDALONE without LLM, using only:

```python
# Rule-based classification signals (NO LLM needed):
1. LOC detection:
   - Extract gold_patch_files from instance["patch"]
   - Check if agent EVER opened/searched/edited any gold_patch file
   - If not → LOC (confidence: 0.8)

2. EDIT detection:
   - Count "not found", "no match", "SyntaxError" in observations
   - If edit_errors >= 2 → EDIT (confidence: 0.7)

3. LOGIC detection:
   - Agent touched gold files + edit applied (no tool error) + test failed
   - → LOGIC (confidence: 0.6)

4. PLAN detection:
   - Agent repeated same action 3+ times (stuck loop)
   - Agent edited files in completely different module than gold patch
   - → PLAN (confidence: 0.5)

5. TEST detection:
   - Agent's final edit is in correct file but test fails with
     assertion about unexpected behavior (not syntax/import)
   - Harder to detect by rules → default to LOGIC, let LLM refine later
```

Script: `scripts/rule_annotate.py` (new, standalone, zero-cost)

**0C. Cross-reference public trajectories with our 200 instances**

- How many of our 200 instances have public trajectories?
- What's the fail/pass distribution per agent?
- Which instances have MULTIPLE agents failing? (richest for analysis)

### What we learn
- Failure type distribution across real agents (Table 1 data)
- Whether our 5-type taxonomy covers >95% of real failures
- Which failure types are most common (guides where to invest API budget)
- Concrete examples for each type (qualitative analysis for paper)

### Go/No-Go for Phase 1
- ✅ GO if: ≥100 failed trajectories from public data overlap with our 200 instances,
  AND rule-based annotation produces recognizable 5-type distribution
- ⚠️ ADJUST if: <100 overlap → supplement with more HF datasets or expand subset
- 🛑 STOP if: Public trajectories are in incompatible format AND can't be parsed
  → Pivot to running our own cheap baseline (Phase 2 early start)

### Deliverables
```
data/public_trajectories/
  ├── swe_agent/{instance_id}.json        # SWE-agent official runs
  ├── agentless/{instance_id}.json        # Agentless predictions  
  ├── openhands/{instance_id}.json        # OpenHands runs
  └── summary.json                        # coverage stats

data/annotations/
  └── rule_based_failure_types.json       # Zero-cost annotations

results/
  └── public_trajectory_analysis.json     # Distribution + coverage
```

---

## ═══════════════════════════════════════
## Phase 1: ZERO-COST Analysis & Paper Skeleton (Day 2: 5/18-5/19 AM)
## Cost: ¥0 | Time: 6-8 hours
## ═══════════════════════════════════════

### What to do

**1A. Full failure taxonomy analysis on public trajectories**

Using rule-based annotations from Phase 0:
- Compute failure type distribution (Figure 2 draft)
- Compute per-repo failure type patterns
- Identify which types are easiest/hardest to detect by rules
- Generate Table 1 (failure types with examples) using REAL examples

**1B. Error cascade analysis (zero-cost, pure analysis)**

For each failed trajectory:
- Identify first_error_step (rule: first step where agent deviates from gold files)
- Compute cascade_length = total_steps - first_error_step
- Compute waste_ratio = wasted_steps / total_steps
- Check if agent attempted self-recovery (did it ever go back to correct files?)

This gives us EXP-006 + EXP-007 data without any API cost!

Script: `scripts/cascade_analysis.py` (new, zero-cost)
```python
# Core cascade metrics (all computable from trajectory + gold patch):
for traj in failed_trajectories:
    gold_files = extract_gold_files(instance["patch"])
    first_error = find_first_deviation(traj["steps"], gold_files)
    cascade_len = len(traj["steps"]) - first_error
    wasted = count_wasted_steps(traj["steps"][first_error:], gold_files)
    waste_ratio = wasted / len(traj["steps"])
```

**1C. Write paper skeleton with placeholder results**

- Draft Introduction (can be finalized now — story doesn't change)
- Draft Related Work (pure literature, no experiments needed)
- Draft §3.2 Failure Type Definitions (from Phase 0 examples)
- Draft §4.1 Cascade Annotation methodology
- Prepare all figure templates (axes labeled, placeholder data)

**1D. Manual expert annotation of 30 trajectories**

Manually read and annotate 30 failed trajectories (diverse selection):
- 6 per expected failure type
- This gives us:
  1. Ground truth for evaluating rule-based classifier accuracy
  2. Few-shot examples for LLM classifier (needed in Phase 2)
  3. Qualitative examples for Figure A1 in appendix
  4. Inter-annotator baseline (you are annotator 1; LLM will be annotator 2)

### What we learn
- Cascade waste ratio (Paper §4.2 core claim: "40-60% of steps wasted")
- Whether self-recovery attempts succeed (RSR metric)
- Rule-based classifier accuracy (vs manual labels on 30 samples)
- Whether failure type distribution matches our predictions

### Go/No-Go for Phase 2
- ✅ GO if: Waste ratio ≥30%, AND rule-based accuracy ≥65% on manual labels,
  AND 5 types each have ≥10 samples
- ⚠️ ADJUST if: Some type has <10 samples → consider merging types (e.g., TEST+PLAN → UNDERSTANDING)
- 🛑 STOP+PIVOT if: Waste ratio <20% → cascades aren't a problem → drop Part 2,
  expand Part 1+3 instead

### Deliverables
```
results/
  ├── failure_distribution_public.json    # Figure 2 data
  ├── cascade_analysis_public.json        # Figure 4 data
  ├── rule_classifier_accuracy.json       # vs manual labels

data/annotations/
  ├── manual_30_samples.json              # Expert annotations
  └── cascade_info_public.json            # Cascade metrics

paper_draft/
  ├── sections_1_2.tex                    # Intro + Related Work
  ├── section_3_skeleton.tex              # Taxonomy skeleton
  └── section_4_skeleton.tex              # Cascade skeleton
```

---

## ═══════════════════════════════════════
## Phase 2: Cheap LLM Validation (Day 2-3: 5/19)
## Cost: ~¥30 (~$4) | Time: 3-4 hours
## ═══════════════════════════════════════

### What to do

**2A. LLM annotation of 50 trajectories (validation batch)**

Purpose: Validate that LLM classifier agrees with rule-based + manual labels.

```bash
python scripts/annotate_failures.py \
  --model gpt-4o-mini \
  --trajectories data/public_trajectories/swe_agent/ \
  --max_calls 50 \
  --output data/annotations/llm_validation_50.json
```

Cost breakdown:
- 50 calls × ~5K input + 500 output tokens each
- GPT-4o-mini: 50 × 5K × ¥0.001/K + 50 × 500 × ¥0.004/K = ¥0.35 + ¥0.10 ≈ **¥0.5**
- Super cheap! Can afford this easily.

**2B. Small-scale scaffolding pilot (5 instances × 5 types)**

The KEY validation: does scaffolding actually help on different failure types?

Pick 5 instances per failure type (25 total) from public failed trajectories.
For each, run ONE strategy (the most promising per type):
- LOC → LOC_C_test_guided
- EDIT → EDIT_A_reread_file (proven in submission1!)
- LOGIC → LOGIC_A_test_analysis
- TEST → TEST_B_test_first  
- PLAN → PLAN_A_step_back

```bash
# 25 instances × 1 strategy each × GPT-4o-mini
# Each retry: ~10 steps × ~5K tokens = ~50K tokens per instance
# 25 × 50K = 1.25M tokens
# Cost: 1M input × ¥0.001 + 0.25M output × ¥0.004 = ¥0.001 + ¥0.001 ≈ ¥2
```

Wait — that's too cheap. The real cost includes the full agent run.
Let me recalculate:
- Each scaffolding run = agent retry with scaffold prompt
- ~10 API calls per instance, each ~5K input + 1K output  
- 25 instances × 10 calls = 250 API calls
- 250 × 5K = 1.25M input tokens, 250 × 1K = 0.25M output
- GPT-4o-mini: 1.25M × ¥0.001/K + 0.25M × ¥0.004/K ≈ **¥2.25**

Plus the 6.5s rate limit: 250 calls × 6.5s = ~27 minutes. Very manageable.

**2C. Pilot control condition (25 instances without scaffold)**

Run the same 25 instances with just "try again" (CONTROL_no_scaffold):
- Same cost as 2B: ~¥2.25
- This gives us the critical comparison: scaffold vs no-scaffold

**Total Phase 2 cost: ~¥5 (incredibly cheap!)**

Actually, let's be generous and budget ¥30 to include some GPT-4.1 pilot:
- 10 instances × GPT-4.1 with scaffold: ~¥15 
- 10 instances × GPT-4.1 without scaffold: ~¥15

### What we learn
- LLM vs rule-based vs manual agreement rate (Table A3)
- **CRITICAL**: Whether scaffolding improves recovery AT ALL on ≥3 types
- Whether the effect varies by type (the core thesis!)
- Whether GPT-4.1 shows similar patterns to GPT-4o-mini
- Preliminary recovery rates per type (Table 3 pilot data)

### Go/No-Go for Phase 3
- ✅ GO if: Scaffolding recovery > control by ≥10% on ≥3 types,
  AND type-specific strategies differ by ≥15% from each other
  → The story works! Scale up.
- ⚠️ ADJUST if: Only 1-2 types show effect → narrow paper scope, 
  focus on those types + explain why others don't respond
- 🛑 STOP+PIVOT if: Scaffolding ≈ control on all types
  → The generalization claim is false → pivot to pure taxonomy + cascade paper
  (drop Part 1 scaffolding, expand Part 2 cascade + Part 3 classifier)

### Deliverables
```
data/annotations/
  └── llm_validation_50.json              # LLM vs rule agreement

data/scaffolding_pilot/
  ├── {failure_type}/{strategy}/gpt-4o-mini/{instance_id}.json
  └── CONTROL/gpt-4o-mini/{instance_id}.json

results/
  ├── pilot_scaffolding_results.json      # Recovery rates per type
  ├── pilot_agreement_stats.json          # 3-way agreement
  └── go_nogo_phase3.md                   # Decision document
```

---

## ═══════════════════════════════════════
## Phase 3: Core Scaffolding Experiments (Day 3-4: 5/19-5/20)
## Cost: ~¥60 (~$8) | Time: 6-8 hours  
## ═══════════════════════════════════════

### Prerequisites
- Phase 2 GO signal confirmed
- Know which types respond to scaffolding (from pilot)
- Have validated LLM classifier accuracy

### What to do

**3A. Full LLM annotation of ALL failed public trajectories**

```bash
python scripts/annotate_failures.py \
  --model gpt-4o-mini \
  --trajectories data/public_trajectories/ \
  --max_calls 300 \
  --output data/annotations/failure_types_public_all.json
```

Cost: ~300 calls × ~5.5K tokens × GPT-4o-mini pricing ≈ **¥3**

**3B. Full scaffolding matrix: 5 types × 3 strategies × GPT-4o-mini**

For each (type, strategy) pair, run on ALL instances of that type:
- Estimated ~40-60 instances per type × 5 types = 200-300 instances
- Each with 3 strategies + 1 control = 4 runs per instance
- But we can be smart: only run 3 strategies for types that showed effect in Phase 2

```bash
# Run all strategies for all types on GPT-4o-mini
python scripts/run_scaffolding.py \
  --model gpt-4o-mini \
  --annotations data/annotations/failure_types_public_all.json \
  --max_calls 2000
```

Cost calculation:
- ~250 failed instances × 4 conditions (3 strategies + control)
- = 1,000 scaffolding runs
- Each run: ~10 API calls × 5K+1K tokens
- Total: 10K API calls × 6K tokens = 60M tokens
- GPT-4o-mini: 50M input × ¥0.001/K + 10M output × ¥0.004/K ≈ **¥50 + ¥40 = ¥90**

Hmm, that's more than budgeted. Optimize:
- Only run 2 strategies per type (best from pilot + 1 alternative) + control
- = 250 × 3 = 750 runs
- Cost: **~¥55**

Actually, re-examining: with 6.5s rate limiting:
- 750 runs × 10 calls = 7,500 calls × 6.5s = ~13.5 hours
- This is too long! 

**Optimized plan:**
- Prioritize: Run BEST strategy per type + control = 250 × 2 = 500 runs
- 5,000 calls × 6.5s = ~9 hours (overnight!)
- Cost: 5,000 × 6K tokens = 30M tokens → **¥30-40**
- Then selectively run 2nd strategy only on types where 1st shows >20% lift

**3C. GPT-4.1 validation (50 instances)**

Run the best strategy + control on 50 instances with GPT-4.1:
- 50 × 2 conditions × 10 calls = 1,000 calls
- 1,000 × 6K tokens = 6M tokens  
- GPT-4.1: 5M × ¥0.014/K + 1M × ¥0.056/K ≈ **¥70 + ¥56 = ¥126**

Too expensive! Reduce to 30 instances:
- 30 × 2 × 10 = 600 calls → 3.6M tokens → **¥75**
- Still expensive. Budget ¥60 max for this.
- 20 instances × 2 conditions: ¥50 — workable.

**Revised Phase 3 budget**:
| Item | Calls | Cost (¥) |
|------|-------|----------|
| LLM annotation (300 trajs) | 300 | 3 |
| GPT-4o-mini scaffolding (best+control, 250 inst) | 5,000 | 35 |
| GPT-4.1 scaffolding (best+control, 20 inst) | 400 | 50 |
| **Total** | 5,700 | **~88** |

Actually let's re-budget. ¥60 is tight. Adjust:
- Skip GPT-4.1 in Phase 3, defer to Phase 4
- GPT-4o-mini only: ¥38 total ← fits

### What we learn
- **Table 3**: Full scaffolding effectiveness matrix (Part 1 core result)
- **Figure 3**: Heatmap of strategy × type recovery rates
- Statistical significance of type-strategy interaction
- Whether capability threshold exists per type

### Go/No-Go for Phase 4
- ✅ GO if: ≥3 types show significant scaffolding effect (p<0.05),
  AND optimal strategy is type-dependent (interaction effect significant)
- ⚠️ ADJUST if: Effect sizes smaller than expected → increase sample sizes for borderline types
- 🛑 STOP: Not applicable (by this point we have a paper either way)

### Deliverables
```
data/annotations/
  └── failure_types_public_all.json       # Full LLM annotations

data/scaffolding/
  ├── LOC/LOC_C_test_guided/gpt-4o-mini/
  ├── EDIT/EDIT_A_reread_file/gpt-4o-mini/
  ├── ...
  └── CONTROL/gpt-4o-mini/

results/
  ├── scaffolding_matrix_gpt-4o-mini.json # Full results
  ├── scaffolding_significance.json       # Statistical tests
  └── figure3_heatmap_data.json           # For plotting
```

---

## ═══════════════════════════════════════
## Phase 4: C&B + Pipeline + Multi-Model (Day 4-5: 5/20-5/21)
## Cost: ~¥80 (~$11) | Time: 8-10 hours
## ═══════════════════════════════════════

### What to do

**4A. C&B-Oracle (zero API cost! uses existing data)**

C&B-Oracle only needs gold patch comparison + trajectory truncation.
We can compute the UPPER BOUND of C&B from existing Phase 3 data:
- For each failed trajectory, find first_error_step (from cascade analysis)
- The scaffolding result from Phase 3 = what happens after backtrack + scaffold
- So: Oracle C&B recovery rate ≈ scaffolding recovery rate when given perfect error detection

This is a BRILLIANT shortcut: Phase 3 data IS the C&B-Oracle data!
Just reframe the results: "Oracle C&B backtracks to first error, then applies type-specific scaffold"

Cost: **¥0** (pure analysis of Phase 3 results)

**4B. C&B-Heuristic (small API cost)**

```bash
python scripts/cb_engine.py \
  --model gpt-4o-mini \
  --mode heuristic \
  --max_calls 200
```

- Uses rule-based error detection (no LLM calls for detection!)
- Only API cost is the agent retry after backtrack
- 200 instances × ~10 calls × 6K tokens = 12M tokens
- GPT-4o-mini: **~¥15**

**4C. C&B-LLM (moderate API cost)**

```bash
python scripts/cb_engine.py \
  --model gpt-4o-mini \
  --mode llm \
  --max_calls 200
```

- LLM error detection: ~5 checkpoints × 200 instances = 1,000 detect calls
- Agent retry: 200 × ~10 = 2,000 calls  
- Total: 3,000 calls × 6K tokens = 18M tokens
- GPT-4o-mini: **~¥20**

**4D. GPT-4.1 core experiments (20-30 instances)**

Run GPT-4.1 on 20 instances to show cross-model generalization:
- Baseline + best scaffold per type + C&B-Heuristic
- 20 × 3 conditions × 10 calls = 600 calls
- GPT-4.1: **~¥50**

**4E. Failure classifier evaluation (from Phase 1+3 data)**

We already have rule-based + LLM annotations + manual labels.
- Train/evaluate the classifier using 80/20 split
- Compute per-type F1 scores → Table 6
- No additional API cost if we use existing annotations!

Cost: **¥0**

**4F. Strategy selector + E2E pipeline (lightweight)**

- Strategy selector is a lookup table from Phase 3 results: ¥0
- E2E evaluation on 30 held-out instances with GPT-4o-mini:
  - 30 × 4 conditions × 10 calls = 1,200 calls → **~¥10**

**Phase 4 budget**:
| Item | Cost (¥) |
|------|----------|
| C&B-Oracle (analysis only) | 0 |
| C&B-Heuristic (200 inst, 4o-mini) | 15 |
| C&B-LLM (200 inst, 4o-mini) | 20 |
| GPT-4.1 validation (20 inst) | 50 |
| E2E pipeline (30 inst) | 10 |
| **Total** | **~¥95** |

Over budget! Optimize:
- Reduce C&B-LLM to 100 instances: ¥10
- Reduce GPT-4.1 to 15 instances: ¥37
- Reduce E2E to 20 instances: ¥7

Revised: **~¥69**. Acceptable.

### What we learn
- **Table 5**: C&B comparison (baseline vs heuristic vs LLM vs oracle)
- **Figure 5**: Resolve rate improvement bar chart
- **Table 6**: Classifier accuracy
- **Table 7**: E2E pipeline results
- Cross-model generalization (GPT-4.1 confirms patterns)

### Deliverables
```
data/cb_oracle/          # Analysis of Phase 3 data
data/cb_heuristic/       # Heuristic C&B runs
data/cb_llm/             # LLM C&B runs

results/
  ├── cb_comparison.json          # Table 5 data
  ├── classifier_eval.json        # Table 6 data
  ├── e2e_pipeline_results.json   # Table 7 data
  ├── gpt41_validation.json       # Cross-model
  └── figure5_cb_bars.json        # Figure 5 data
```

---

## ═══════════════════════════════════════
## Phase 5: Polish + Supplementary (Day 5-6: 5/21-5/22)
## Cost: ~¥30 | Time: 4-6 hours
## ═══════════════════════════════════════

### What to do

**5A. Statistical significance tests (¥0)**
- Bootstrap CIs for all main results
- McNemar's test for pairwise comparisons
- Cohen's h effect sizes
- p-values for Table 3, 5, 7

**5B. Ablation study (¥15)**
- 3 ablation conditions × 30 instances × GPT-4o-mini
- Random classifier, best-single strategy, no backtrack

**5C. DeepSeek V4 cross-family validation (¥15)**
- Run 20 instances with DeepSeek V4
- Baseline + best scaffold + C&B-Heuristic
- Addresses "is this OpenAI-specific?" concern

**5D. Generate all figures (¥0)**
- Figure 2: Failure distribution (from Phase 1 data)
- Figure 3: Scaffolding heatmap (from Phase 3 data)
- Figure 4: Cascade distribution (from Phase 1 data)
- Figure 5: C&B comparison (from Phase 4 data)
- Figure 1 + 6: Diagrams (hand-drawn / tikz)

### Deliverables
```
results/
  ├── significance_tests.json
  ├── ablation_results.json
  └── deepseek_validation.json

figures/
  ├── figure1_overview.pdf
  ├── figure2_failure_dist.pdf
  ├── figure3_scaffolding_heatmap.pdf
  ├── figure4_cascade_dist.pdf
  ├── figure5_cb_comparison.pdf
  └── figure6_pipeline.pdf
```

---

## ═══════════════════════════════════════
## Total Budget Summary
## ═══════════════════════════════════════

| Phase | What | API Cost (¥) | Time | Day |
|-------|------|-------------|------|-----|
| **0** | Download public trajectories + rule-based annotation | **0** | 4-6h | 5/17-18 |
| **1** | Cascade analysis + paper skeleton + manual annotation | **0** | 6-8h | 5/18-19 |
| **2** | LLM validation (50) + scaffolding pilot (25) | **~5-30** | 3-4h | 5/19 |
| **3** | Full scaffolding matrix (GPT-4o-mini, 250 inst) | **~38** | 8-10h | 5/19-20 |
| **4** | C&B experiments + GPT-4.1 + pipeline | **~69** | 8-10h | 5/20-21 |
| **5** | Ablation + DeepSeek + significance + figures | **~30** | 4-6h | 5/21-22 |
| **Writing** | Full paper drafting + polish | **0** | 16-20h | 5/22-24 |
| **Buffer** | Reruns, fixes, additional data | **~30** | — | — |
| **TOTAL** | | **~¥200** | ~60h | 8 days |

vs. Original plan: ~$500 = ¥3,600. **We save 94% by using public data first!**

---

## ═══════════════════════════════════════
## Day-by-Day Schedule
## ═══════════════════════════════════════

### Day 1 (5/17 PM - 5/18 AM): Phase 0
- [ ] Write `scripts/download_public_trajectories.py`
- [ ] Download SWE-agent, Agentless, OpenHands predictions from HuggingFace/GitHub
- [ ] Normalize trajectories to common format
- [ ] Run rule-based annotator on all public failures
- [ ] Compute instance overlap with our 200-instance subset
- [ ] **Checkpoint**: How many usable failed trajectories do we have?

### Day 2 (5/18): Phase 1 + Start Phase 2
- [ ] Run cascade analysis on public trajectories
- [ ] Manually annotate 30 trajectories (expert labels)
- [ ] Compute rule-based classifier accuracy vs manual
- [ ] Draft Introduction, Related Work
- [ ] Start Phase 2: LLM annotation validation batch (50 calls)

### Day 3 (5/19): Phase 2 + Start Phase 3
- [ ] Complete Phase 2: scaffolding pilot (25 instances)
- [ ] **GO/NO-GO DECISION**: Does scaffolding work across types?
- [ ] If GO: Start Phase 3 overnight batch (GPT-4o-mini scaffolding matrix)
- [ ] Draft §3 (Failure Taxonomy) with Phase 1 data

### Day 4 (5/20): Phase 3 complete + Phase 4 start
- [ ] Analyze Phase 3 results (scaffolding matrix)
- [ ] Start C&B experiments (heuristic + LLM modes)
- [ ] Run GPT-4.1 validation (20 instances)
- [ ] Draft §4 (Cascade + C&B)

### Day 5 (5/21): Phase 4 complete + Phase 5
- [ ] Complete C&B experiments
- [ ] Run E2E pipeline evaluation
- [ ] Ablation + DeepSeek validation
- [ ] Statistical tests
- [ ] Draft §5 (Automated Pipeline)

### Day 6 (5/22): Writing
- [ ] Complete all figures
- [ ] Write Discussion + Limitations + Conclusion
- [ ] Polish all sections
- [ ] Internal review pass

### Day 7 (5/23): Polish + Submit
- [ ] Final proofread
- [ ] Check all numbers match between text and tables
- [ ] Compile LaTeX
- [ ] Any last-minute reruns if numbers look off

### Day 8 (5/24-25): Buffer + Submit
- [ ] Fix any issues found
- [ ] Submit before AoE deadline

---

## ═══════════════════════════════════════
## Risk Mitigation: Anti-白跑 Strategies
## ═══════════════════════════════════════

### Risk 1: Public trajectories unavailable or incompatible format
**Mitigation**: Start with ONE source (SWE-agent predictions are most standard).
If format issues, just parse the model_patch + pass/fail status — we don't need
full step-by-step trajectories for taxonomy analysis, just:
- Which files were edited (from model_patch diff)
- Whether it resolved (from eval results)
- Compare edited files vs gold files → LOC classification
**Fallback**: Run our own GPT-4o-mini baseline on 50 instances (¥5, 2 hours)

### Risk 2: Scaffolding doesn't generalize (Phase 2 NO-GO)
**Mitigation**: Pivot to "taxonomy + cascade" paper (still novel, still publishable):
- Part 1 becomes: Pure failure taxonomy analysis
- Part 2 becomes: Cascade quantification (the 40-60% waste number is independently valuable)
- Part 3 becomes: Failure classifier only (no strategy selection)
- This is a shorter but still solid EMNLP paper

### Risk 3: Budget overrun
**Mitigation**: 
- GPT-4o-mini is ¥0.001/K input — incredibly cheap
- ALL expensive experiments (GPT-4.1, DeepSeek) are OPTIONAL supplements
- Core paper can be written entirely with GPT-4o-mini data + public trajectories
- If budget is tight, skip GPT-5.5 ceiling analysis entirely

### Risk 4: Rate limiting / Venus API issues
**Mitigation**:
- 6.5s delay already built into all scripts
- Run overnight batches (Phase 3, 4)
- Incremental saves protect against crashes
- Can split large batches across multiple sessions

### Risk 5: Not enough time for writing
**Mitigation**:
- Paper skeleton drafted in Phase 1 (Day 2), before experiments finish
- Results fill into pre-prepared templates
- Introduction and Related Work are experiment-independent
- If time-pressed, submit with smaller experiment set and expand in revision

---

## ═══════════════════════════════════════
## Key Insight: Why Public Trajectories Are Gold
## ═══════════════════════════════════════

The paper's EXPERIMENTS.md budgeted $193 (¥1,390) for EXP-003 alone (baseline trajectory collection).
That's 70% of the budget, just to GET failed trajectories to analyze.

But those same trajectories already exist publicly:
- SWE-bench has a public leaderboard with dozens of submissions
- Most top submissions publish their predictions (pass/fail + patches)
- Several publish FULL trajectories (SWE-agent, OpenHands)

By using public data:
1. ¥0 for what was budgeted at ¥1,390 (EXP-003)
2. ¥0 for cascade analysis (EXP-006 + EXP-007)  
3. ¥0 for rule-based taxonomy validation
4. Only spend on what MUST be run fresh: scaffolding interventions + C&B

The scaffolding and C&B experiments inherently require API calls (you're CHANGING the agent's behavior). 
But everything ANALYTICAL can be done for free on public data.

**This is the key asymmetry**: 
Analysis is free. Intervention costs money. Do ALL analysis first.

---

## ═══════════════════════════════════════  
## Script Implementation Priority
## ═══════════════════════════════════════

### Priority 1 (Today, Phase 0):
```
scripts/download_public_trajectories.py   # NEW: fetch HF/GitHub public data
scripts/rule_annotate.py                  # NEW: standalone rule-based annotator
scripts/cascade_analysis.py               # NEW: zero-cost cascade metrics
```

### Priority 2 (Phase 2-3, already exist, minor tweaks):
```
scripts/annotate_failures.py              # EXISTS: add mode to work on public trajs
scripts/run_scaffolding.py                # EXISTS: ready to use
```

### Priority 3 (Phase 4):
```
scripts/cb_engine.py                      # EXISTS: ready to use
scripts/run_agent.py                      # EXISTS: only needed if running own baseline
```

---

*Plan created: 2026-05-17*
*Philosophy: 每一分钱都在验证之后再花。先免费分析，再小额验证，最后全量投入。*
