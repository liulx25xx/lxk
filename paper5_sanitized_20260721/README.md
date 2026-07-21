# When Does RLVR Beat SFT? A Controlled Multi-Domain Study

> **AAAI-27 revision status (2026-07-21):** the manuscript and older experiment
> trackers contain provisional or superseded claims. Do not use them as the
> source of truth for new runs. Start from
> [`docs/aaai27/00_START_HERE.md`](docs/aaai27/00_START_HERE.md). The workflow is
> intentionally split into **Phase A: experiments first** and **Phase B: paper
> revision after results are frozen**.

EMNLP 2026 Submission

## Overview

Systematic comparison of **RLVR (GRPO)** vs **SFT** vs **DPO** across 6 domains under controlled conditions (same model, same data, same compute). Identifies the **RLVR benefit frontier** -- when does RL training add value over supervised fine-tuning?

## Domains

| Domain | Benchmarks | Reward Type |
|--------|-----------|-------------|
| Math | GSM8K + MATH-500 | Numerical exact match |
| Science | ScienceQA + ARC-Challenge | MCQ answer match |
| Law | LegalBench | Binary/MCQ match |
| Medicine | MedQA (USMLE) | MCQ answer match |
| Code | HumanEval + MBPP | Test case execution |
| Commonsense | ARC-Easy + HellaSwag | MCQ answer match |

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### Full pipeline
```bash
bash src/scripts/run_all.sh
```

### Individual steps
```bash
# 1. Data preparation
python src/data/prepare_datasets.py --output_dir data/raw
python src/data/create_splits.py --raw_dir data/raw --output_dir data/splits
python src/data/format_sft.py --input_dir data/splits --output_dir data/formatted/sft
python src/data/format_rlvr.py --input_dir data/splits --output_dir data/formatted/rlvr

# 2. Training (example: math domain, 5K samples)
torchrun --nproc_per_node=8 src/training/train_sft.py \
    --data_path data/formatted/sft/math/5000/train.jsonl \
    --output_dir outputs/sft/math/5000

torchrun --nproc_per_node=8 src/training/train_grpo.py \
    --data_path data/formatted/rlvr/math/5000/train.jsonl \
    --output_dir outputs/grpo/math/5000

# 3. Evaluation
python src/eval/evaluate.py \
    --model_path outputs/sft/math/5000/final \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --test_dir data/raw --domains math

# 4. Analysis
python src/analysis/plot_frontier.py --results_dir results
python src/analysis/compute_statistics.py --results_dir results
python src/analysis/generate_tables.py --results_dir results
```

## Fair Comparison Design

All methods use identical settings for fair comparison:

| Setting | Value |
|---------|-------|
| Base model | Qwen2.5-7B-Instruct |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA targets | All linear layers |
| Precision | BF16 |
| Compute budget | SFT 3 epochs ~ GRPO 4000 steps (G=8) |

## Experiment Axes

1. **Domain effect**: Which domains benefit most from RLVR?
2. **Data size**: 500 / 2,000 / 5,000 / 20,000 training instances
3. **Difficulty**: Easy / Medium / Hard splits within each domain
4. **OOD generalization**: Train on one difficulty/dataset, eval on another

## Project Structure

```
paper2/
  src/
    data/
      prepare_datasets.py    # Download all 6 domains from HuggingFace
      create_splits.py       # Create size (500-20K) and difficulty splits
      format_sft.py          # Format for SFT (question + CoT + answer)
      format_rlvr.py         # Format for GRPO (question-only prompts)
      format_dpo.py          # Generate DPO preference pairs
    reward/
      rewards.py             # All reward functions (math, mcq, code, binary)
    training/
      train_sft.py           # SFT with LoRA + DeepSpeed
      train_grpo.py          # GRPO with verifiable rewards
      train_dpo.py           # DPO from preference pairs
      configs/               # YAML/JSON configs
    eval/
      evaluate.py            # Evaluate any checkpoint on any benchmark
      metrics.py             # Accuracy, pass@k, confidence intervals
    analysis/
      plot_frontier.py       # "RLVR Benefit Frontier" heatmap + all figures
      compute_statistics.py  # McNemar's test, bootstrap CI, effect sizes
      generate_tables.py     # LaTeX table generation
    scripts/
      run_all.sh             # Master pipeline
      run_domain_comparison.sh
      run_data_scaling.sh
      run_difficulty.sh
  data/                      # Raw and formatted datasets (generated)
  outputs/                   # Trained models (generated)
  results/                   # Evaluation results (generated)
  figures/                   # Paper figures (generated)
  tables/                    # LaTeX tables (generated)
```

## Hardware Requirements

- Recommended: 8x H200 (80GB) GPUs
- Each training run: ~8-12 hours
- Total experiment time: ~5 days with parallelization
