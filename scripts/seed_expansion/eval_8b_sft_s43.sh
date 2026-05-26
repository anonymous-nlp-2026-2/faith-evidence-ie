#!/bin/bash
# Eval: 8B SFT bf16 seed=43 (D076 protocol: bs=4, no-quantize, seed=42)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SFT_ADAPTER="./sft_output_qwen3_8b_s43"

if [ ! -f "${SFT_ADAPTER}/adapter_config.json" ]; then
    echo "ERROR: SFT adapter not found at ${SFT_ADAPTER}/adapter_config.json"
    echo "Training may not have completed yet."
    exit 1
fi

python -m freige.eval.inference \
    --model_path Qwen/Qwen3-8B \
    --sft_adapter "${SFT_ADAPTER}" \
    --data_path data/docred \
    --split dev \
    --output_dir eval_results/8b_sft_s43 \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
