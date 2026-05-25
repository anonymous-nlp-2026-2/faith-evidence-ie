#!/bin/bash
# GRPO bf16 G=4 step 100 checkpoint evaluation
# Usage: CUDA_VISIBLE_DEVICES=X bash scripts/eval_grpo_step100.sh
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

cd /workspace

CKPT_DIR="/workspace/grpo_bf16_g4/checkpoint-100"
BASE_MODEL="/workspace/models/Qwen3-4B"
SFT_ADAPTER="/workspace/sft_output"
DATA_PATH="/workspace/data/docred"
OUTPUT_DIR="/workspace/eval_results/grpo_bf16_g4_step100"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint-100 not found at ${CKPT_DIR}"
    echo "Current checkpoints:"
    ls -d /workspace/grpo_bf16_g4/checkpoint-* 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "=== GRPO bf16 G=4 Step 100 Eval ==="
echo "  Checkpoint: ${CKPT_DIR}"
echo "  Base:       ${BASE_MODEL}"
echo "  SFT:        ${SFT_ADAPTER}"
echo "  Output:     ${OUTPUT_DIR}"
echo ""

python -m freige.eval.inference \
    --model_path "${CKPT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --sft_adapter "${SFT_ADAPTER}" \
    --data_path "${DATA_PATH}" \
    --split dev \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --no-quantize

echo ""
echo "=== GRPO Step 100 Eval Done ==="
