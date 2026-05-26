#!/bin/bash
# No-evidence eval for 8B RSFT s42 (evidence tax measurement)
# Model: Qwen3-8B + SFT s42 adapter + RSFT s42 adapter, with --no_evidence
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path ./rsft_output_qwen3_8b_k1 \
    --base_model Qwen/Qwen3-8B \
    --sft_adapter ./sft_output_qwen3_8b_bf16/ \
    --data_path data/docred \
    --output_dir ./eval_outputs/eval_8b_rsft_s42_no_evidence \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --no_evidence \
    --seed 42
