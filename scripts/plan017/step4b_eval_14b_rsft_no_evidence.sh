#!/bin/bash
# Step 4b: D076 eval for 14B RSFT (no-evidence variant — evidence tax measurement)
# Must pass --no_evidence to use NO_EVIDENCE_SYSTEM_PROMPT (C005)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-2,3}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RSFT_ADAPTER="./rsft_output_14b"
SFT_ADAPTER="./sft_output_14b"
BASE_MODEL="Qwen/Qwen3-14B"
OUTPUT_DIR="eval_results/plan017_14b_rsft_no_evidence_eval"

# --- Pre-flight checks ---
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

echo "=== Step 4b: Eval 14B RSFT (no-evidence) ==="
echo "Base:    $BASE_MODEL"
echo "SFT:     $SFT_ADAPTER"
echo "RSFT:    $RSFT_ADAPTER"
echo "Output:  $OUTPUT_DIR"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"

python -m freige.eval.inference \
    --model_path "${RSFT_ADAPTER}" \
    --base_model "${BASE_MODEL}" \
    --sft_adapter "${SFT_ADAPTER}" \
    --data_path data/docred \
    --split dev \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --no_evidence \
    --seed 42

echo "=== Eval Complete ==="
