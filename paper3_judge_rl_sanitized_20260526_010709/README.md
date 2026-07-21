# When Accuracy Follows Position

Anonymous research artifact for **“When Accuracy Follows Position: Diagnosing Training-Induced Shortcuts in LLM Judges.”**

## What this repository studies

Pairwise judge training can contain a deterministic position confound: a naive conversion from `chosen`/`rejected` records places the preferred response in slot A on every example. A model can then improve original-order accuracy by learning “choose A” rather than comparing response quality.

The paper audits this behavior by evaluating every test pair in both response orders. Its primary diagnostics are:

- **Original-order accuracy:** accuracy in the confounded orientation.
- **Swapped-order accuracy:** accuracy after exchanging the two physical responses.
- **Order-averaged accuracy:** the equal-weight mean of the two orientation accuracies.
- **Position-swap consistency:** whether the verdict flips appropriately after the responses are exchanged.
- **First-position rate:** how often the judge selects the first-listed response across both orientations.

The main intervention balances preferred responses across positions during training. On the primary Qwen2.5-7B setting, unbalanced full GRPO reaches 94.7% original-order accuracy but 74.7% order-averaged accuracy and 58.7% consistency. Balanced full GRPO reaches 82.6% order-averaged accuracy and 84.0% consistency; balanced SFT reaches 90.2% order-averaged accuracy.

## Important interpretation notes

- The fixed-order conversion is a **controlled training construction**, not a claim that RewardBench or preference datasets inherently use that order.
- The GRPO mode historically named `acc_consist` uses a **decisiveness proxy** that penalizes ties; it does not compute paired swap consistency during training.
- The mode historically named `acc_calib` uses a Brier-shaped score with a fixed fallback confidence when the model emits no confidence. The paper therefore calls it a **fixed-confidence proxy**, not a calibration result.
- The random train/validation split is recorded for reproducibility, but the validation portion is not used for checkpoint selection or tuning in the reported runs.

## Repository layout

```text
paper/                 LaTeX source and figures
scripts/               Data conversion, training, evaluation, and aggregation
results/               Raw evaluation records and generated summaries
results/local_stats/   Recomputed tables, diagnostics, and uncertainty checks
output/pdf/            Stable compiled submission PDF
artifact/              Anonymous submission-artifact staging area
```

Historical experiment notes outside the staged artifact are retained as internal provenance and may use obsolete terminology. Only `artifact/blackboxnlp2026/` is intended for external release.

## Recompute reported summaries

The aggregation script reads saved `eval_results.json` files and does not require a GPU:

```bash
python scripts/summarize_existing_results.py \
  --results-dir results \
  --out-dir results/local_stats
```

Key outputs include `selected_group_summary.csv`, `uncertainty_checks.csv`, `dose_response_summary.csv`, and their Markdown counterparts.

## Build the paper

From `paper/`:

```bash
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

## Training and evaluation

The scripts use Hugging Face model identifiers by default and respect standard cache environment variables. Typical entry points are:

```bash
python scripts/prepare_data.py
python scripts/prepare_ratio_data.py --help
python scripts/train_judge_sft.py --help
python scripts/train_judge_dpo.py --help
python scripts/train_judge_grpo.py --help
python scripts/eval_judge.py --help
```

Model checkpoints and licensed source datasets are not included. Training the 7B models requires a suitable CUDA environment; recomputing the paper's tables from saved predictions is CPU-only.

## Anonymity and release scope

The external artifact contains no author identity, private host, or private filesystem path. Before release, use the staged artifact archive under `output/artifact/`, not the full working repository.
