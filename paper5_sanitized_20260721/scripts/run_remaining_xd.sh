#!/bin/bash
# Remaining cross-domain evals (GPUs 5-7 now, GPU 4 after current process finishes)
set -e

export HF_HOME=/path/to/workspace/cache/huggingface
export HF_HUB_CACHE=/path/to/workspace/cache/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/workspace/cache/huggingface/hub
export HF_TOKEN=<HF_TOKEN_REDACTED>
export TRITON_CACHE_DIR=/path/to/workspace/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/workspace/cache/torch_inductor
export TMPDIR=/path/to/workspace/cache/tmp

PYTHON=/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python
PROJECT=/path/to/workspace/project/emnlp/paper5
SCRIPT=$PROJECT/src/eval/eval_ood.py
OUTDIR=$PROJECT/eval_results/cross_domain
LOGDIR=$PROJECT/logs

cd $PROJECT

run_eval() {
    local gpu=$1
    local domain=$2
    local model_path=$3
    local tag=$4
    echo "[$(date)] GPU=$gpu domain=$domain tag=$tag"
    $PYTHON $SCRIPT --gpu $gpu --domain $domain \
        --model_path $model_path \
        --model_tag $tag \
        --gpu_mem 0.85 \
        --output_dir $OUTDIR \
        --cleanup_merge >> $LOGDIR/xd_${tag}.log 2>&1
    echo "[$(date)] DONE: $tag"
}

# --- IMMEDIATE: GPUs 5,6,7 (GPU 4 busy with med_grpo_s42) ---
echo "=== Round 1: GPUs 5-7 (3 evals) ==="
run_eval 5 medicine $PROJECT/outputs/science/grpo/FIXED_seed789_n2000_lr2e5/final/ xd_sci_grpo_s789 &
run_eval 6 science $PROJECT/outputs/medicine/grpo/FIXED_seed123_n2000_lr2e5/final/ xd_med_grpo_s123 &
run_eval 7 science $PROJECT/outputs/medicine/grpo/FIXED_seed456_n2000_lr2e5/final/ xd_med_grpo_s456 &
wait
echo "=== Round 1 done ==="

# --- Wait for GPU 4 to free up ---
echo "Waiting for GPU 4..."
while nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 4 | grep -qv "0 MiB"; do
    sleep 10
done
echo "GPU 4 free!"

# --- Round 2: All 4 GPUs ---
echo "=== Round 2: Medicine GRPO seed789 + Math GRPO (4 seeds) → Science ==="
run_eval 4 science $PROJECT/outputs/medicine/grpo/FIXED_seed789_n2000_lr2e5/final/ xd_med_grpo_s789 &
run_eval 5 science $PROJECT/outputs/math/grpo/FIXED_seed42_gsm8k_n2000_lr2e5/final/ xd_math_grpo_s42 &
run_eval 6 science $PROJECT/outputs/math/grpo/FIXED_seed123_gsm8k_n2000_lr2e5/final/ xd_math_grpo_s123 &
run_eval 7 science $PROJECT/outputs/math/grpo/FIXED_seed456_gsm8k_n2000_lr2e5/final/ xd_math_grpo_s456 &
wait
echo "=== Round 2 done ==="

echo "=== Round 3: Math GRPO seed789 + SFT Science→Medicine ==="
run_eval 4 science $PROJECT/outputs/math/grpo/FIXED_seed789_gsm8k_n2000_lr2e5/final/ xd_math_grpo_s789 &
run_eval 5 medicine $PROJECT/outputs/science/sft/seed42_n2000/final/ xd_sci_sft_s42 &
run_eval 6 medicine $PROJECT/outputs/science/sft/seed123_n2000/final/ xd_sci_sft_s123 &
run_eval 7 medicine $PROJECT/outputs/science/sft/seed456_n2000/final/ xd_sci_sft_s456 &
wait
echo "=== Round 3 done ==="

echo "=== Round 4: SFT Medicine→Science (4 seeds) ==="
run_eval 4 science $PROJECT/outputs/medicine/sft/seed42_n2000/final/ xd_med_sft_s42 &
run_eval 5 science $PROJECT/outputs/medicine/sft/seed123_n2000/final/ xd_med_sft_s123 &
run_eval 6 science $PROJECT/outputs/medicine/sft/seed456_n2000/final/ xd_med_sft_s456 &
run_eval 7 science $PROJECT/outputs/medicine/sft/seed789_n2000/final/ xd_med_sft_s789 &
wait
echo "=== Round 4 done ==="

echo "=== Round 5: SFT Math→Science (3 seeds) ==="
run_eval 4 science $PROJECT/outputs/math/sft/seed42_n2000/final/ xd_math_sft_s42 &
run_eval 5 science $PROJECT/outputs/math/sft/seed123_n2000/final/ xd_math_sft_s123 &
run_eval 6 science $PROJECT/outputs/math/sft/seed456_n2000/final/ xd_math_sft_s456 &
wait
echo "=== Round 5 done ==="

echo ""
echo "====== ALL CROSS-DOMAIN EVALS COMPLETE ======"
echo "Results:"
ls $OUTDIR/*.json | wc -l
echo "json files in $OUTDIR"
echo ""
echo "Summary:"
for f in $OUTDIR/science_xd_*.json $OUTDIR/medicine_xd_*.json; do
    if [ -f "$f" ]; then
        tag=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d[\"model_tag\"]}: {d[\"accuracy\"]:.4f}')")
        echo "  $tag"
    fi
done
