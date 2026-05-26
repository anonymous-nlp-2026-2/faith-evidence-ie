#!/bin/bash
# No-evidence eval for 1.7B RSFT s42 with max_new_tokens=2048
# Rerun to rule out truncation artifact: original 1024 had 59.2% truncation rate (583/985)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path ./rsft_output_qwen3_1_7b_k1 \
    --base_model Qwen/Qwen3-1.7B \
    --sft_adapter ./sft_output_qwen3_1_7b_bf16 \
    --data_path data/docred \
    --output_dir ./eval_outputs/eval_1_7b_rsft_no_evidence_2048 \
    --batch_size 4 \
    --max_new_tokens 2048 \
    --no-quantize \
    --no_evidence \
    --seed 42
