#!/bin/bash
# Eval: 8B RSFT k=1 seed=44 (D076 protocol: bs=4, no-quantize, seed=42)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path /workspace/rsft_output_qwen3_8b_k1_s44 \
    --base_model /workspace/models/Qwen/Qwen3-8B \
    --sft_adapter /workspace/sft_output_qwen3_8b_bf16 \
    --data_path /workspace/data/docred \
    --output_dir /workspace/eval_results/rsft_8b_k1_s44_eval \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
