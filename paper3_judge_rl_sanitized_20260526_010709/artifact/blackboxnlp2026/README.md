# Anonymous Artifact: When Accuracy Follows Position

This package supports the paper **“When Accuracy Follows Position: Diagnosing Training-Induced Shortcuts in LLM Judges.”** It is designed for anonymous review and CPU-only verification of the reported tables from saved per-example predictions.

## Fast path: reproduce the summaries

From the artifact root, run:

```bash
python3 scripts/summarize_existing_results.py \
  --results-dir results \
  --out-dir reproduced_stats
```

The summarizer uses only the Python standard library. It recomputes original-order accuracy, swapped-order accuracy, order-averaged accuracy, position-swap consistency, first-position rate, position-bias diagnostics, domain slices, and the reported local bootstrap checks. Compare `reproduced_stats/` with `results/local_stats/`.

Expected headline values on Qwen2.5-7B are:

| Setting | Original acc. | Swapped acc. | Order-avg. acc. | Consistency |
|---|---:|---:|---:|---:|
| Baseline | 80.2 | 75.3 | 77.7 | 81.5 |
| Full GRPO, unbalanced | 94.7 | 54.8 | 74.7 | 58.7 |
| Full GRPO, balanced | 83.7 | 81.4 | 82.6 | 84.0 |
| SFT, balanced | 91.3 | 89.1 | 90.2 | 87.1 |

Multi-seed GRPO rows are means. The paper tables report seed standard deviations; local bootstrap intervals in `uncertainty_checks.csv` are descriptive because the number of seeds is small.

## Artifact contents

```text
paper/                 Main and appendix LaTeX sources, bibliography, shared figure style, figure source/PDF
scripts/               Data conversion, training, evaluation, aggregation
results/**/eval_results.json
                       Saved per-example judge outputs
results/**/metrics.json
                       Stored aggregates with machine paths redacted
results/local_stats/   Reference summaries generated from those outputs
```

The per-example result JSON files contain identifiers, verdicts, rationales, and derived fields, but not the licensed source prompts or candidate answers. Stored aggregate files are included for integrity checks after replacing machine-specific checkpoint paths with a redaction marker. Model checkpoints are excluded because of size. Upstream datasets and base models must be obtained under their respective licenses.

## Metric contract

- **Original accuracy:** verdict correctness in the fixed original orientation.
- **Swapped accuracy:** correctness after physically exchanging the responses and flipping the gold A/B label.
- **Order-averaged accuracy:** `(original accuracy + swapped accuracy) / 2`.
- **Consistency:** A/B verdicts must flip after swapping; C/C is treated as self-consistent.
- **First-position rate:** fraction of all original and swapped verdicts selecting the first slot.

The fixed-order conversion is a controlled construction, not an assertion that RewardBench inherently uses this ordering.

## Prepare data

`scripts/prepare_data.py` loads the upstream RewardBench data, shuffles with seed 42, and records the 70/15/15 train/reserved-validation/test split used by the project. The validation slice was not saved or used for checkpoint selection in the reported runs. The original training orientation places `chosen` in slot A; the paired swap file physically exchanges the responses and assigns gold B.

Generate intermediate confound-ratio datasets with:

```bash
python3 scripts/prepare_ratio_data.py \
  --train-data data/train/rewardbench_train.json \
  --swap-data data/train/rewardbench_train_swap.json \
  --output-dir data/train \
  --ratios 0.60 0.75 0.80 0.90 0.95
```

The intermediate ratios are single-seed, non-monotonic diagnostics; they are not presented as a smooth dose-response result.

## Train and evaluate

Install the packages listed in `requirements-training.txt` in a CUDA-compatible environment, then inspect each command's `--help`. Representative entry points are:

```bash
python3 scripts/train_judge_sft.py --help
python3 scripts/train_judge_dpo.py --help
python3 scripts/train_judge_grpo.py --help
python3 scripts/eval_judge.py --help
```

The legacy GRPO CLI names are retained so saved run commands remain interpretable:

- `acc_consist` adds a decisiveness proxy, not paired swap consistency.
- `acc_calib` adds a Brier-shaped fixed-confidence proxy, not an empirical calibration objective.
- `full` adds both proxies to label accuracy.

Exact training-library versions were not preserved in the saved run metadata, so the training requirements are intentionally not presented as an exact lockfile. The saved predictions and CPU-only summary path are the primary verification route.

## Build the paper

With a recent TeX Live installation:

```bash
cd paper
latexmk -pdf figure1.tex
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

`figure1.tex` regenerates the vector mechanism figure and reads the same `figure_style.tex` used by every PGFPlots chart. The main source conditionally includes the appendix for the complete submission PDF.
