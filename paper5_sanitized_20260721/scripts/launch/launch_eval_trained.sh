#!/bin/bash
# Launch parallel evaluation of trained models on <REDACTED_IP>
# GPUs 0-3, 5-7 are free (GPU 4 is busy)
#
# Strategy: each GPU evaluates a batch of models sequentially
# (vLLM needs full GPU, can't share)

export HF_HOME=/path/to/workspace/cache/huggingface
export TRITON_CACHE_DIR=/path/to/workspace/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/workspace/cache/torch_inductor
export TMPDIR=/path/to/workspace/cache/tmp
export WANDB_MODE=disabled

PYTHON=/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python
SCRIPT=/path/to/workspace/project/emnlp/paper5/src/eval/eval_trained_models.py
LOGDIR=/path/to/workspace/project/emnlp/paper5/logs

mkdir -p $LOGDIR

# GPU 0: Medicine models (6 models: 3 SFT + 3 GRPO)
echo "Launching GPU 0: Medicine models"
nohup $PYTHON $SCRIPT --gpu 0 --domain medicine --skip_existing \
    > $LOGDIR/eval_trained_gpu0_medicine.log 2>&1 &

# GPU 1: Math SFT models (6 models)
echo "Launching GPU 1: Math SFT"
nohup $PYTHON $SCRIPT --gpu 1 --domain math --method sft --skip_existing \
    > $LOGDIR/eval_trained_gpu1_math_sft.log 2>&1 &

# GPU 2: Math GRPO models (5 models)
echo "Launching GPU 2: Math GRPO"
nohup $PYTHON $SCRIPT --gpu 2 --domain math --method grpo --skip_existing \
    > $LOGDIR/eval_trained_gpu2_math_grpo.log 2>&1 &

# GPU 3: Science SFT (2 models) + Law (2 models) + Commonsense (1 model)
echo "Launching GPU 3: Science"
nohup $PYTHON $SCRIPT --gpu 3 --domain science --skip_existing \
    > $LOGDIR/eval_trained_gpu3_science.log 2>&1 &

# GPU 5: Law models
echo "Launching GPU 5: Law"
nohup $PYTHON $SCRIPT --gpu 5 --domain law --skip_existing \
    > $LOGDIR/eval_trained_gpu5_law.log 2>&1 &

# GPU 6: Commonsense + Code
echo "Launching GPU 6: Commonsense"
nohup $PYTHON $SCRIPT --gpu 6 --domain commonsense --skip_existing \
    > $LOGDIR/eval_trained_gpu6_commonsense.log 2>&1 &

# GPU 7: Code GRPO (3 models)
echo "Launching GPU 7: Code"
nohup $PYTHON $SCRIPT --gpu 7 --domain code --skip_existing \
    > $LOGDIR/eval_trained_gpu7_code.log 2>&1 &

echo "All eval jobs launched. Monitor with:"
echo "  tail -f $LOGDIR/eval_trained_gpu*.log"
echo "  nvidia-smi on <REDACTED_IP>"
