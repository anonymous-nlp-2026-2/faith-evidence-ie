#!/bin/bash
# Eval: D112 4B RSFT bf16 (SFT s44 base + CED k=1 RSFT, D076 protocol)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RSFT_ADAPTER="./rsft_output_4b_sft_s44_bf16"
SFT_ADAPTER="./sft_output_seed44"

if [ ! -f "${RSFT_ADAPTER}/adapter_config.json" ]; then
    echo "ERROR: RSFT adapter not found at ${RSFT_ADAPTER}/adapter_config.json"
    echo "Training may not have completed yet. Latest checkpoints:"
    ls -d "${RSFT_ADAPTER}"/checkpoint-* 2>/dev/null || echo "  (none)"
    exit 1
fi

if [ ! -f "${SFT_ADAPTER}/adapter_config.json" ]; then
    echo "ERROR: SFT adapter not found at ${SFT_ADAPTER}/adapter_config.json"
    exit 1
fi

python -m freige.eval.inference \
    --model_path "${RSFT_ADAPTER}" \
    --base_model Qwen/Qwen3-4B \
    --sft_adapter "${SFT_ADAPTER}" \
    --data_path data/docred \
    --split dev \
    --output_dir eval_results/d112_rsft_bf16 \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
