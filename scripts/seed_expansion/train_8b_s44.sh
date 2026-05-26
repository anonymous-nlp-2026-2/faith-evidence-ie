#!/bin/bash
# 8B RSFT k=1 seed=44
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.training.rsft_trainer \
    --base_model Qwen/Qwen3-8B \
    --sft_adapter ./sft_output_qwen3_8b_bf16 \
    --rsft_data_path ./rsft_scored_qwen3_8b_k1/rsft_train.jsonl \
    --output_dir ./rsft_output_qwen3_8b_k1_s44 \
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
    --wandb_run_name rsft_8b_k1_s44
