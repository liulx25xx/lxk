#!/bin/bash
# Cross-domain eval script for Paper 5 merge
# Runs sequentially per batch (4 parallel on GPUs 4-7), waits, then next batch

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

mkdir -p $OUTDIR $LOGDIR

run_eval() {
    local gpu=$1
    local domain=$2
    local model_path=$3
    local tag=$4
    local logfile=$LOGDIR/xd_${tag}.log
    
    echo "[$(date)] Starting: GPU=$gpu domain=$domain tag=$tag"
    cd $PROJECT
    $PYTHON $SCRIPT --gpu $gpu --domain $domain \
        --model_path $model_path \
        --model_tag $tag \
        --gpu_mem 0.85 \
        --output_dir $OUTDIR \
        --cleanup_merge > $logfile 2>&1
    echo "[$(date)] Done: $tag (see $logfile)"
}

echo "====== BATCH 1: Science-trained GRPO → Medicine OOD (seed789 only, others done) ======"
run_eval 4 medicine $PROJECT/outputs/science/grpo/FIXED_seed789_n2000_lr2e5/final/ xd_sci_grpo_s789 &
PID1=$!

echo "====== BATCH 2: Medicine-trained GRPO → Science OOD (4 seeds) ======"
# GPU 4 is used by batch1 seed789, use 5-7 + wait for 4
run_eval 5 science $PROJECT/outputs/medicine/grpo/FIXED_seed123_n2000_lr2e5/final/ xd_med_grpo_s123 &
run_eval 6 science $PROJECT/outputs/medicine/grpo/FIXED_seed456_n2000_lr2e5/final/ xd_med_grpo_s456 &
run_eval 7 science $PROJECT/outputs/medicine/grpo/FIXED_seed789_n2000_lr2e5/final/ xd_med_grpo_s789 &

wait $PID1
echo "[$(date)] Batch 1 seed789 done, launching med_grpo_s42 on GPU 4"
run_eval 4 science $PROJECT/outputs/medicine/grpo/FIXED_seed42_n2000_lr2e5/final/ xd_med_grpo_s42 &

wait
echo "[$(date)] Batch 2 complete"

echo "====== BATCH 3: Math-trained GRPO → Science OOD (4 seeds) ======"
run_eval 4 science $PROJECT/outputs/math/grpo/FIXED_seed42_gsm8k_n2000_lr2e5/final/ xd_math_grpo_s42 &
run_eval 5 science $PROJECT/outputs/math/grpo/FIXED_seed123_gsm8k_n2000_lr2e5/final/ xd_math_grpo_s123 &
run_eval 6 science $PROJECT/outputs/math/grpo/FIXED_seed456_gsm8k_n2000_lr2e5/final/ xd_math_grpo_s456 &
run_eval 7 science $PROJECT/outputs/math/grpo/FIXED_seed789_gsm8k_n2000_lr2e5/final/ xd_math_grpo_s789 &
wait
echo "[$(date)] Batch 3 complete"

echo "====== BATCH 4: Science-trained SFT → Medicine OOD (3 seeds) ======"
run_eval 4 medicine $PROJECT/outputs/science/sft/seed42_n2000/final/ xd_sci_sft_s42 &
run_eval 5 medicine $PROJECT/outputs/science/sft/seed123_n2000/final/ xd_sci_sft_s123 &
run_eval 6 medicine $PROJECT/outputs/science/sft/seed456_n2000/final/ xd_sci_sft_s456 &
wait
echo "[$(date)] Batch 4 complete"

echo "====== BATCH 5: Medicine-trained SFT → Science OOD (4 seeds) ======"
run_eval 4 science $PROJECT/outputs/medicine/sft/seed42_n2000/final/ xd_med_sft_s42 &
run_eval 5 science $PROJECT/outputs/medicine/sft/seed123_n2000/final/ xd_med_sft_s123 &
run_eval 6 science $PROJECT/outputs/medicine/sft/seed456_n2000/final/ xd_med_sft_s456 &
run_eval 7 science $PROJECT/outputs/medicine/sft/seed789_n2000/final/ xd_med_sft_s789 &
wait
echo "[$(date)] Batch 5 complete"

echo "====== BATCH 6: Math-trained SFT → Science OOD (3 seeds) ======"
run_eval 4 science $PROJECT/outputs/math/sft/seed42_n2000/final/ xd_math_sft_s42 &
run_eval 5 science $PROJECT/outputs/math/sft/seed123_n2000/final/ xd_math_sft_s123 &
run_eval 6 science $PROJECT/outputs/math/sft/seed456_n2000/final/ xd_math_sft_s456 &
wait
echo "[$(date)] Batch 6 complete"

echo ""
echo "====== ALL DONE ======"
echo "Results in: $OUTDIR"
ls $OUTDIR/*.json | wc -l
echo "json files produced"
