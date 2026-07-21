#!/bin/bash
# Launch parallel evaluation across multiple GPUs on multiple servers.
# Each GPU gets a range of models to evaluate serially.

set -e

export HF_HOME=/path/to/workspace/cache/huggingface
export TRITON_CACHE_DIR=/path/to/workspace/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/workspace/cache/torch_inductor
export TMPDIR=/path/to/workspace/cache/tmp

PYTHON=/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python
SCRIPT=/path/to/workspace/project/emnlp/paper5/scripts/eval/eval_batch_v2.py
LOG_DIR=/path/to/workspace/project/emnlp/paper5/logs/eval_v2

mkdir -p $LOG_DIR

# Server, GPU, model indices to evaluate
# 25 models total, distribute across available GPUs
# Priority models (0-4) go to the fastest/most reliable GPUs

# 244: GPUs 1,2,3,5,6 → models 0-4 (priority) then 5-9
# 82: GPUs 0,2,4,5,6,7 → models 10-15
# 182: GPUs 1,2,3,4,5,6,7 → models 16-24 (but we only need 9 more)

run_eval() {
    local server=$1
    local gpu=$2
    local start=$3
    local end=$4
    local tag="${server}_gpu${gpu}_idx${start}_${end}"
    
    echo "Launching on $server GPU $gpu: models $start-$((end-1))"
    ssh $server "nohup bash -c '
        export HF_HOME=/path/to/workspace/cache/huggingface
        export TRITON_CACHE_DIR=/path/to/workspace/cache/triton
        export TORCHINDUCTOR_CACHE_DIR=/path/to/workspace/cache/torch_inductor
        export TMPDIR=/path/to/workspace/cache/tmp
        export CUDA_VISIBLE_DEVICES=$gpu
        cd /path/to/workspace/project/emnlp/paper5
        $PYTHON $SCRIPT --start $start --end $end
    ' > $LOG_DIR/${tag}.log 2>&1 &"
}

# Priority models on 244 (closest/fastest)
run_eval <REDACTED_IP> 1 0 1    # fullft (most critical)
run_eval <REDACTED_IP> 2 1 2    # lr5e6
run_eval <REDACTED_IP> 3 2 3    # dapo
run_eval <REDACTED_IP> 5 3 4    # opd
run_eval <REDACTED_IP> 6 4 5    # mathfull

# Medicine + math models on 82
run_eval <REDACTED_IP> 0 5 7     # med grpo s123 n2000 + n500
run_eval <REDACTED_IP> 2 7 9     # med grpo s42 n2000 + lr1e5
run_eval <REDACTED_IP> 4 9 11    # med grpo lr1e6 + v2
run_eval <REDACTED_IP> 5 11 13   # math grpo gsm8k n2000 + n5000
run_eval <REDACTED_IP> 6 13 15   # math sft s123 n100 + n500
run_eval <REDACTED_IP> 7 15 17   # science grpo s123 n2000_v2 + n5000

# Science + law + commonsense on 182
run_eval <REDACTED_IP> 1 17 19  # science grpo s42 n2000 + n5000
run_eval <REDACTED_IP> 2 19 20  # science grpo lr1e6
run_eval <REDACTED_IP> 3 20 21  # science sft s123 n2000
run_eval <REDACTED_IP> 4 21 22  # law grpo s123
run_eval <REDACTED_IP> 5 22 23  # law grpo lr5e6
run_eval <REDACTED_IP> 6 23 24  # commonsense grpo
run_eval <REDACTED_IP> 7 24 25  # commonsense sft s123

echo ""
echo "Launched 18 eval jobs across 3 servers (25 models total)."
echo "Monitor: tail -f $LOG_DIR/*.log"
echo "Check results: ls -la /path/to/workspace/project/emnlp/paper5/eval_results/trained/*.json | wc -l"
