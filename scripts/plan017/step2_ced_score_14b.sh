#!/bin/bash
# Step 2: CED scoring for 14B RSFT generations
# Input: rsft_generations_14b/ (from plan017_14b_rsft_generate)
# Output: rsft_scored_14b/ (rsft_train.jsonl + rsft_scores.jsonl + rsft_report.json)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

INPUT_DIR="./rsft_generations_14b"
OUTPUT_DIR="./rsft_scored_14b"
NLI_MODEL="cross-encoder/nli-deberta-v3-base"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "Ensure plan017_14b_rsft_generate has completed."
    exit 1
fi

echo "=== Step 2: CED Scoring (14B) ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

python -m freige.training.rsft_score_filter \
    --input_dir "$INPUT_DIR" \
    --output_path "$OUTPUT_DIR" \
    --nli_model_path "$NLI_MODEL" \
    --selection_strategy top_k \
    --top_k 1 \
    --scoring_mode ced

echo "=== CED Scoring Complete ==="
echo "Training data: $OUTPUT_DIR/rsft_train.jsonl"
wc -l "$OUTPUT_DIR/rsft_train.jsonl"
cat "$OUTPUT_DIR/rsft_report.json"
