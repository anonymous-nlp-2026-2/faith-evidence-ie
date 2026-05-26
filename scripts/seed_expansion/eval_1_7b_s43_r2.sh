#!/bin/bash
# D076 eval for 1.7B RSFT s43_r2 (Method A)
# Model: Qwen3-1.7B + SFT s42 adapter + RSFT s43 adapter
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path ./rsft_output_qwen3_1_7b_k1_s43 \
    --base_model Qwen/Qwen3-1.7B \
    --sft_adapter ./sft_output_qwen3_1_7b_bf16 \
    --data_path data/docred \
    --output_dir eval_results/rsft_1_7b_s43_r2_eval \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
