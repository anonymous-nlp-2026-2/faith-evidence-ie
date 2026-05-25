#!/bin/bash
# Eval: 1.7B SFT seed=43 (D076 protocol: bs=4, no-quantize, seed=42)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path /workspace/models/Qwen/Qwen3-1.7B \
    --sft_adapter /workspace/sft_output_qwen3_1_7b_bf16_seed43 \
    --data_path /workspace/data/docred \
    --output_dir /workspace/eval_results/sft_1_7b_s43_eval \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
