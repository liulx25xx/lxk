# Merge Plan: Paper 1 (OPD Generalization) → Paper 5 (One Recipe Does Not Fit All)

## Goal
Add cross-domain transfer instability finding from Paper 1 as a new section in Paper 5.

**Key claim to add:** GRPO's cross-domain transfer is seed-dependent (high variance across seeds), while SFT's transfer is stable. This extends Paper 5's thesis: "not only does recipe choice matter, but even with the right recipe, GRPO's transfer reliability is SEED-DEPENDENT."

## Paper 1's Finding (from `/path/to/workspace/project/emnlp/auto/opd_generalization/`)
- 9 GRPO seeds trained on math (GSM8K) → evaluated on ARC-Challenge, MMLU
- GRPO cross-domain std: 5.8 (ARC), 4.5 (MMLU) → bimodal clustering
- SFT cross-domain std: <0.2 on same benchmarks
- **58× variance ratio** (GRPO/SFT)
- Some seeds generalize well, others don't → unreliable transfer

## What We Have in Paper 5

### Checkpoints (all Qwen2.5-7B-Instruct + LoRA, lr=2e-5, n=2000)

| Domain | GRPO Seeds | SFT Seeds |
|--------|-----------|-----------|
| Medicine | 42, 123, 456, 789 | 42, 123, 456, 789 |
| Science | 42, 123, 456, 789 | 42, 123, 456 |
| Math (GSM8K) | 42, 123, 456, 789 | 42, 123, 456 |

**Checkpoint paths:**
- GRPO: `outputs/{domain}/grpo/FIXED_seed{S}_n2000_lr2e5/final/`
  - Math: `outputs/math/grpo/FIXED_seed{S}_gsm8k_n2000_lr2e5/final/`
- SFT: `outputs/{domain}/sft/seed{S}_n2000/final/`

### OOD Benchmarks Available (`data/ood/`)

| File | Source | N samples | Notes |
|------|--------|-----------|-------|
| math_ood.jsonl | MATH-500 | 500 | Math OOD (trains on GSM8K) |
| science_ood.jsonl | ARC-Challenge | 1172 | Science OOD |
| medicine_ood.jsonl | MMLU-Med | 945 | Medicine OOD |
| law_ood.jsonl | MMLU-Law | 1763 | Law OOD |
| commonsense_ood.jsonl | WinoGrande | 1267 | Commonsense OOD |

### Cross-Domain Eval Design

For the merge, we evaluate each checkpoint on **domains it was NOT trained on**:
- Medicine-trained → eval on: math, science, law, commonsense (4 OOD domains)
- Science-trained → eval on: math, medicine, law, commonsense (4 OOD domains)
- Math-trained → eval on: science, medicine, law, commonsense (4 OOD domains)

**Full matrix:** 3 training domains × 4 seeds × 2 algorithms × 4 OOD benchmarks = 96 evals (GRPO: 48, SFT: 44 with 3 seeds on science/math)

**Priority subset (to show variance):** Focus on 2 OOD benchmarks per training domain:
- Medicine-trained → eval on math (MATH-500) + science (ARC-C)
- Science-trained → eval on math (MATH-500) + medicine (MMLU-Med)
- Math-trained → eval on science (ARC-C) + medicine (MMLU-Med)

**Priority matrix:** 3 domains × 4 seeds × 2 algos × 2 OOD = 48 evals (24 GRPO + 22 SFT)

## Eval Script

**Script:** `src/eval/eval_ood.py`

**Usage:**
```bash
PYTHON=/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python

$PYTHON src/eval/eval_ood.py \
    --gpu 4 \
    --domain medicine \
    --model_path /path/to/workspace/project/emnlp/paper5/outputs/science/grpo/FIXED_seed42_n2000_lr2e5/final/ \
    --model_tag science_grpo_seed42 \
    --gpu_mem 0.85
```

**Key args:**
- `--gpu`: GPU index (use 4-7 on 119.14)
- `--domain`: OOD benchmark to evaluate ON (math/science/medicine/law/commonsense)
- `--model_path`: path to adapter (ABSOLUTE PATH required)
- `--model_tag`: short name for output file
- `--gpu_mem`: vLLM memory utilization (default 0.50, can use 0.85 with full GPU)
- `--cleanup_merge`: delete merged model after eval (saves disk)

**Output:** `eval_results/cross_domain/{domain}_{model_tag}_ood.json`
(We'll use `--output_dir eval_results/cross_domain/` to separate from existing single-seed OOD results)

## Launch Commands

### Phase 1: GRPO Multi-Seed Cross-Domain (Priority)

Run 4 evals in parallel on GPUs 4-7:

```bash
# Science-trained GRPO → eval on Medicine (MMLU-Med)
# 4 seeds × 1 benchmark = 4 evals, one per GPU

cd /path/to/workspace/project/emnlp/paper5
PYTHON=/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python
export HF_HOME=/path/to/workspace/cache/huggingface
export HF_HUB_CACHE=/path/to/workspace/cache/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/workspace/cache/huggingface/hub
export HF_TOKEN=<HF_TOKEN_REDACTED>

# Batch 1: Science-trained GRPO → Medicine OOD
$PYTHON src/eval/eval_ood.py --gpu 4 --domain medicine --model_path /path/to/workspace/project/emnlp/paper5/outputs/science/grpo/FIXED_seed42_n2000_lr2e5/final/ --model_tag xd_sci_grpo_s42 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 5 --domain medicine --model_path /path/to/workspace/project/emnlp/paper5/outputs/science/grpo/FIXED_seed123_n2000_lr2e5/final/ --model_tag xd_sci_grpo_s123 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 6 --domain medicine --model_path /path/to/workspace/project/emnlp/paper5/outputs/science/grpo/FIXED_seed456_n2000_lr2e5/final/ --model_tag xd_sci_grpo_s456 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 7 --domain medicine --model_path /path/to/workspace/project/emnlp/paper5/outputs/science/grpo/FIXED_seed789_n2000_lr2e5/final/ --model_tag xd_sci_grpo_s789 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
wait

# Batch 2: Medicine-trained GRPO → Science OOD (ARC-C)
$PYTHON src/eval/eval_ood.py --gpu 4 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/medicine/grpo/FIXED_seed42_n2000_lr2e5/final/ --model_tag xd_med_grpo_s42 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 5 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/medicine/grpo/FIXED_seed123_n2000_lr2e5/final/ --model_tag xd_med_grpo_s123 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 6 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/medicine/grpo/FIXED_seed456_n2000_lr2e5/final/ --model_tag xd_med_grpo_s456 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 7 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/medicine/grpo/FIXED_seed789_n2000_lr2e5/final/ --model_tag xd_med_grpo_s789 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
wait

# Batch 3: Math-trained GRPO → Science OOD (ARC-C)
$PYTHON src/eval/eval_ood.py --gpu 4 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/math/grpo/FIXED_seed42_gsm8k_n2000_lr2e5/final/ --model_tag xd_math_grpo_s42 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 5 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/math/grpo/FIXED_seed123_gsm8k_n2000_lr2e5/final/ --model_tag xd_math_grpo_s123 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 6 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/math/grpo/FIXED_seed456_gsm8k_n2000_lr2e5/final/ --model_tag xd_math_grpo_s456 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 7 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/math/grpo/FIXED_seed789_gsm8k_n2000_lr2e5/final/ --model_tag xd_math_grpo_s789 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
wait
```

### Phase 2: SFT Multi-Seed Cross-Domain (Control)

```bash
# Batch 4: Science-trained SFT → Medicine OOD
$PYTHON src/eval/eval_ood.py --gpu 4 --domain medicine --model_path /path/to/workspace/project/emnlp/paper5/outputs/science/sft/seed42_n2000/final/ --model_tag xd_sci_sft_s42 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 5 --domain medicine --model_path /path/to/workspace/project/emnlp/paper5/outputs/science/sft/seed123_n2000/final/ --model_tag xd_sci_sft_s123 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 6 --domain medicine --model_path /path/to/workspace/project/emnlp/paper5/outputs/science/sft/seed456_n2000/final/ --model_tag xd_sci_sft_s456 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
wait

# Batch 5: Medicine-trained SFT → Science OOD
$PYTHON src/eval/eval_ood.py --gpu 4 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/medicine/sft/seed42_n2000/final/ --model_tag xd_med_sft_s42 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 5 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/medicine/sft/seed123_n2000/final/ --model_tag xd_med_sft_s123 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 6 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/medicine/sft/seed456_n2000/final/ --model_tag xd_med_sft_s456 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 7 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/medicine/sft/seed789_n2000/final/ --model_tag xd_med_sft_s789 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
wait

# Batch 6: Math-trained SFT → Science OOD
$PYTHON src/eval/eval_ood.py --gpu 4 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/math/sft/seed42_n2000/final/ --model_tag xd_math_sft_s42 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 5 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/math/sft/seed123_n2000/final/ --model_tag xd_math_sft_s123 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
$PYTHON src/eval/eval_ood.py --gpu 6 --domain science --model_path /path/to/workspace/project/emnlp/paper5/outputs/math/sft/seed456_n2000/final/ --model_tag xd_math_sft_s456 --gpu_mem 0.85 --output_dir eval_results/cross_domain/ --cleanup_merge &
wait
```

### Time Estimate
- Each eval: merge (~60s) + vLLM load (~30s) + generate (~2-5 min for 500-1200 samples) = ~5-8 min
- With 4 GPUs in parallel: 6 batches × ~8 min = ~48 min total (well under 1h)

## Integration into Paper 5

### New Section: §4.5 "Cross-Domain Transfer Instability"

**Narrative position:** After showing GRPO's in-domain advantages (§4.1-4.4), reveal a critical limitation:

> "The preceding sections demonstrate GRPO's advantages under optimal configuration. A natural question arises: do these in-domain gains transfer to unseen domains? We evaluate each trained model on benchmarks outside its training domain across 4 seeds."

**Key table/figure:**

| Training Domain | Algorithm | OOD Benchmark | Mean Acc | Std | Range |
|----------------|-----------|---------------|----------|-----|-------|
| Science | GRPO | MMLU-Med | X.X | Y.Y | Z.Z |
| Science | SFT | MMLU-Med | X.X | Y.Y | Z.Z |
| Medicine | GRPO | ARC-C | X.X | Y.Y | Z.Z |
| Medicine | SFT | ARC-C | X.X | Y.Y | Z.Z |
| Math | GRPO | ARC-C | X.X | Y.Y | Z.Z |
| Math | SFT | ARC-C | X.X | Y.Y | Z.Z |

**Expected finding:** GRPO std >> SFT std across all domain pairs → "GRPO transfer is seed-dependent"

**Contribution upgrade:** Paper 5's thesis strengthens from "recipe choice matters" to "recipe choice matters, AND GRPO's advantages come with transfer unreliability that SFT avoids."

### Figure Suggestion
Box plot showing GRPO vs SFT cross-domain accuracy distributions (4 seeds each) → visually shows GRPO's wider spread.

### Connection to Paper 1
Paper 1's 9-seed finding (std=5.8 on ARC) was with a DIFFERENT training setup. Paper 5's replication with its own checkpoints (4 seeds, 3 domains) makes the finding MORE ROBUST because:
1. Different training data sizes/domains
2. Same conclusion across both setups
3. Paper 5 can additionally show which domains exhibit MORE vs LESS transfer instability

## Missing / To Verify
- [x] OOD data files exist: all 5 domains present
- [x] GRPO lr2e5 checkpoints: 4 seeds × 3 domains = 12 adapters confirmed
- [x] SFT n2000 checkpoints: 3-4 seeds × 3 domains = 10-11 adapters confirmed
- [x] Eval script functional: `eval_ood.py` handles adapter merge + vLLM eval
- [x] cross_domain results dir exists (empty, ready for results)
- [ ] Need to verify eval_ood.py works on 119.14 (vLLM + PEFT installed in openrlhf env)

## Status
- **Phase 1 launched:** 2026-05-26 (GRPO cross-domain on 119.14 GPUs 4-7)
- **Results expected:** ~48 min after launch
