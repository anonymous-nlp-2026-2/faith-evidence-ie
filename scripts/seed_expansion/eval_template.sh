#!/bin/bash
# Eval template - set MODEL_PATH, BASE_MODEL, SFT_ADAPTER, OUTPUT_DIR, SEED before sourcing
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --sft_adapter "${SFT_ADAPTER}" \
    --data_path data/docred \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed "${SEED}"
