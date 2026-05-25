#!/bin/bash
set -e

BASE_MODEL="/workspace/models/Qwen3-4B"
SFT_ADAPTER="/workspace/sft_output"
DATA_PATH="/workspace/data/docred"
OUTPUT_BASE="/workspace/eval_results/grpo_extract"

cd /workspace

# r4 checkpoints (early steps)
for STEP in 10 20 30; do
    CKPT_DIR="/workspace/grpo_extract_r4/checkpoint-${STEP}"
    OUTPUT_DIR="${OUTPUT_BASE}_r4_step${STEP}"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "=== SKIP r4 step ${STEP}: checkpoint not found ==="
        continue
    fi

    echo "========================================"
    echo "=== Evaluating r4 step ${STEP} ==="
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

    echo "=== r4 step ${STEP} DONE ==="
    echo ""
done

# r3 checkpoints (late steps)
for STEP in 680 700 720 740 756; do
    CKPT_DIR="/workspace/grpo_extract_r3/checkpoint-${STEP}"
    OUTPUT_DIR="${OUTPUT_BASE}_r3_step${STEP}"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "=== SKIP r3 step ${STEP}: checkpoint not found ==="
        continue
    fi

    echo "========================================"
    echo "=== Evaluating r3 step ${STEP} ==="
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

    echo "=== r3 step ${STEP} DONE ==="
    echo ""
done

echo "========================================"
echo "=== All extract reward evaluations complete ==="
echo "========================================"
