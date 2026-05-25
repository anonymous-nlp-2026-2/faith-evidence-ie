#!/bin/bash
# LLaMA-3.1-8B RSFT k=1 seed=44
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.training.rsft_trainer \
    --base_model /workspace/models/meta-llama/Meta-Llama-3.1-8B \
    --sft_adapter /workspace/sft_output_llama_3_1_8b \
    --rsft_data_path /workspace/rsft_scored_llama_3_1_8b_k1/rsft_train.jsonl \
    --output_dir /workspace/rsft_output_llama_3_1_8b_k1_s44 \
    --seed 44 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 100 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 2048 \
    --bf16 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --wandb_run_name rsft_llama_k1_s44
