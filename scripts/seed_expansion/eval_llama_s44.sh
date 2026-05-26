#!/bin/bash
# Eval: LLaMA-3.1-8B RSFT k=1 seed=44 (D076 protocol: bs=4, no-quantize, seed=42)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.eval.inference \
    --model_path ./rsft_output_llama_3_1_8b_k1_s44 \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --sft_adapter ./sft_output_llama_3_1_8b \
    --data_path data/docred \
    --output_dir eval_results/rsft_llama_k1_s44_eval \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
