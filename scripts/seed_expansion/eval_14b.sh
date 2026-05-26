#!/bin/bash
# Eval: 14B SFT (D076 protocol: bs=4, no-quantize, seed=42)
# 14B bf16 = ~28GB, needs 2-GPU with device_map="auto"
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0,1}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path Qwen/Qwen3-14B \
    --sft_adapter ./sft_output_qwen3_14b \
    --data_path data/docred \
    --output_dir eval_results/sft_14b_eval \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
