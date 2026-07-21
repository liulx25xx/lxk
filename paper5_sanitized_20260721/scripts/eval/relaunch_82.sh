#!/bin/bash
# Re-launch eval jobs on 82 with proper HF cache settings

LOG_DIR=/path/to/workspace/project/emnlp/paper5/logs/eval_v2
PYTHON=/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python
SCRIPT=/path/to/workspace/project/emnlp/paper5/scripts/eval/eval_batch_v2.py

relaunch() {
    local gpu=$1 start=$2 end=$3
    local tag="10.3.28.82_gpu${gpu}_idx${start}_${end}"
    echo "Re-launching 82 GPU $gpu: models $start-$((end-1))"
    ssh <REDACTED_IP> "nohup bash -c 'export HF_HOME=/path/to/workspace/cache/huggingface && export HF_HUB_CACHE=/path/to/workspace/cache/huggingface/hub && export TRANSFORMERS_CACHE=/path/to/workspace/cache/huggingface/hub && export TRITON_CACHE_DIR=/path/to/workspace/cache/triton && export TORCHINDUCTOR_CACHE_DIR=/path/to/workspace/cache/torch_inductor && export TMPDIR=/path/to/workspace/cache/tmp && export CUDA_VISIBLE_DEVICES=$gpu && cd /path/to/workspace/project/emnlp/paper5 && $PYTHON $SCRIPT --start $start --end $end' > $LOG_DIR/${tag}.log 2>&1 &"
}

relaunch 0 5 7
relaunch 2 7 9
relaunch 4 9 11
relaunch 5 11 13
relaunch 6 13 15
relaunch 7 15 17

echo "Done re-launching 82 jobs"
