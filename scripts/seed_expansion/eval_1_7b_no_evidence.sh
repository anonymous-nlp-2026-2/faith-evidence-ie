#!/bin/bash
# No-evidence eval for 1.7B RSFT s42 (evidence tax measurement)
# Model: Qwen3-1.7B + SFT s42 adapter + RSFT s42 adapter, with --no_evidence
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path /workspace/rsft_output_qwen3_1_7b_k1 \
    --base_model /workspace/models/Qwen/Qwen3-1.7B \
    --sft_adapter /workspace/sft_output_qwen3_1_7b_bf16 \
    --data_path /workspace/data/docred \
    --output_dir /workspace/eval_outputs/eval_1_7b_rsft_no_evidence \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --no_evidence \
    --seed 42
