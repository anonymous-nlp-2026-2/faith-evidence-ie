#!/bin/bash
set -e

CKPT_BASE="./grpo_ced_g8_bf16"
BASE_MODEL="Qwen/Qwen3-4B"
SFT_ADAPTER="./sft_output"
DATA_PATH="data/docred"
OUTPUT_BASE="eval_results/plan_011_bf16_g8"

cd .

for STEP in 30 50 70; do
    CKPT_DIR="${CKPT_BASE}/checkpoint-${STEP}"
    OUTPUT_DIR="${OUTPUT_BASE}_step${STEP}"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "=== SKIP step ${STEP}: checkpoint not found at ${CKPT_DIR} ==="
        continue
    fi

    echo "========================================"
    echo "=== Evaluating step ${STEP} ==="
    echo "=== Checkpoint: ${CKPT_DIR} ==="
    echo "=== Output: ${OUTPUT_DIR} ==="
    echo "========================================"

    python -m freige.eval.inference \
        --model_path "$CKPT_DIR" \
        --sft_adapter "$SFT_ADAPTER" \
        --base_model "$BASE_MODEL" \
        --data_path "$DATA_PATH" \
        --split dev \
        --batch_size 16 \
        --max_new_tokens 1024 \
        --seed 42 \
        --output_dir "$OUTPUT_DIR" \
        --no-quantize

    echo "=== Step ${STEP} DONE ==="
    echo ""
done

echo "========================================"
echo "=== All evaluations complete ==="
echo "========================================"
