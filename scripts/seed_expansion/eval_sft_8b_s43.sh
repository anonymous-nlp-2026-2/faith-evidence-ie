#!/bin/bash
# Eval: 8B SFT seed=43 (D076 protocol: bs=4, no-quantize, eval seed=42)
# SFT model uses merged adapter, no --sft_adapter needed
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path ./sft_output_8b_seed43 \
    --base_model Qwen/Qwen3-8B \
    --data_path data/docred \
    --output_dir eval_results/sft_8b_s43_eval \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
