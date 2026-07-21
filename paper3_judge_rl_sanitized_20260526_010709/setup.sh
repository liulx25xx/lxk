#!/bin/bash
# Paper 3: Judge RL — 环境搭建脚本
# 在任一服务器执行一次即可 (NFS 共享)
# ⚠️ 确保没有其他训练在使用 judge_rl 环境！

set -e

echo "=== Paper 3: Judge RL Setup ==="
echo "Installing dependencies into judge_rl conda env..."

PIP=/path/to/env/miniconda3/envs/judge_rl/bin/pip
PYTHON=/path/to/env/miniconda3/envs/judge_rl/bin/python

# 设置 cache 到 NFS
export HF_HOME=/path/to/cache/huggingface
export TRITON_CACHE_DIR=/path/to/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/cache/torch_inductor
export TMPDIR=/path/to/cache/tmp

# 确保 cache 目录存在
mkdir -p $HF_HOME $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMPDIR

# 安装核心依赖
echo "[1/3] Installing PyTorch..."
$PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "[2/3] Installing ML packages..."
$PIP install \
    transformers>=4.45 \
    trl>=1.0.0 \
    accelerate \
    datasets \
    peft \
    pandas \
    numpy \
    scikit-learn

echo "[3/3] Installing vLLM (for evaluation)..."
$PIP install vllm

# 验证安装
echo ""
echo "=== Verification ==="
$PYTHON -c "
import torch
import transformers
import trl
import peft
import datasets
print(f'torch:          {torch.__version__}')
print(f'transformers:   {transformers.__version__}')
print(f'trl:            {trl.__version__}')
print(f'peft:           {peft.__version__}')
print(f'datasets:       {datasets.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    print(f'GPU memory:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print()
print('All imports OK ✓')
"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Run: $PYTHON /path/to/paper3_judge_rl/scripts/prepare_data.py"
echo "  2. Run pilot: see EXPERIMENTS.md Step 2"
