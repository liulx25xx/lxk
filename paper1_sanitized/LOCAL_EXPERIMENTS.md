# Local Offline Experiments

Generated entirely from stored per-instance outputs. No API, GPU, or
repository container was used.

## 1. Leave-one-project-out scaffold selection

Across 10 project-held-out folds, the selected scaffold scores 2.229 versus 1.792 for control. The paired delta is +0.438 [+0.333, +0.674] (n=96, projects=10) using a project-clustered bootstrap.

Per-category held-out deltas:

- EDIT: +1.000 [+0.792, +1.192] (n=28, projects=9)
- LOC: +0.133 [-0.857, +0.333] (n=30, projects=5)
- LOGIC: +0.100 [-0.083, +0.476] (n=30, projects=6)
- PLAN: +0.875 [+0.800, +1.000] (n=8, projects=3)

Selection counts across project folds:

- EDIT: {'EDIT_A_reread_file': 10}
- LOC: {'LOC_B_reread_issue': 8, 'LOC_A_broaden_search': 2}
- LOGIC: {'LOGIC_B_minimal_fix': 9, 'LOGIC_A_test_analysis': 1}
- PLAN: {'PLAN_A_step_back': 10}

## 2. Proxy-component sensitivity

Aggregate paired deltas for the manuscript-selected scaffolds:

- Full 0--3 proxy: +0.500 [+0.393, +0.750] (n=96, projects=10)
- Without type-lexical relevance: +0.000 [-0.088, +0.204] (n=96, projects=10)
- Without disclosed-file hit: +0.365 [+0.273, +0.605] (n=96, projects=10)
- Actionable only: -0.135 [-0.213, +0.059] (n=96, projects=10)

The full-proxy gain disappears when the type-specific relevance term
is removed. This indicates that the large reported matched gains are
primarily lexical/context-following effects, not evidence of repair.

## 3. Stored LLM-judge sensitivity

- Regex proxy on the paired judged subset: +0.649 [+0.400, +0.846] (n=37, projects=9)
- Blinded LLM judge on the same subset: +0.135 [-0.158, +0.240] (n=37, projects=9)

The LLM-judge estimate is substantially smaller and its interval
includes zero; the judged subset is also incomplete and should not be
treated as a replacement primary metric.

## 4. Paired cross-category transfer

Using each instance's own control, 10/12 full-proxy means are negative and 11/12 remain negative without the relevance term.
Project-clustered intervals lie fully below zero for 3/12 full-proxy pairs and 3/12 no-relevance pairs.

This paired analysis supersedes the earlier 9/12 count, which mixed
cross-type subsets with category-wide control means.

## Interpretation for the paper

- Supported locally: scaffold effects vary by operational category;
  mismatched scaffolds often reduce the immediate proxy.
- Not supported locally: large matched gains under evaluator-independent
  next-action quality, online routing, or end-to-end repair.
- Still requires new runs: prompts without gold category/files and
  repository-level continuation to test actual recovery.

## Paper figure regeneration

The beautified figures retain the stored values and project-clustered
confidence intervals; only their presentation changes.

```bash
python3 scripts/plot_experiment_summary.py
python3 scripts/plot_cross_model_heatmap.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf main_neurips.tex
```

- `figure2_experiment_summary.{pdf,svg,png}` combines evaluator sensitivity
  with paired cross-category transfer.
- `figure3_cross_model_heatmap.{pdf,svg,png}` uses a continuous diverging scale
  centered at zero.
- `results/cross_model_effects.csv` is the processed matrix used by Figure 3.
- `FIGURE_STYLE.md` records the shared palette, semantic mappings, active
  assets, and legacy-file boundary.
