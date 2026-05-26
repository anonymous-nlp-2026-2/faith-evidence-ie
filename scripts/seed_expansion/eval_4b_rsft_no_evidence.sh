#!/bin/bash
# No-evidence eval for 4B RSFT s44 (evidence tax measurement)
# Model: Qwen3-4B + SFT s42 adapter + RSFT s44 adapter, with --no_evidence
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path ./rsft_output_4b_sft_s44_bf16 \
    --base_model Qwen/Qwen3-4B \
    --sft_adapter ./sft_output_seed44/ \
    --data_path data/docred \
    --output_dir ./eval_outputs/eval_4b_rsft_s44_no_evidence \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --no_evidence \
    --seed 42
