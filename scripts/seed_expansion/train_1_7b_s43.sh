#!/bin/bash
# 1.7B RSFT k=1 seed=43
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.training.rsft_trainer \
    --base_model /workspace/models/Qwen/Qwen3-1.7B \
    --sft_adapter /workspace/sft_output_qwen3_1_7b_bf16 \
    --rsft_data_path /workspace/freige/outputs/rsft_scored_1_7b_k1/rsft_train.jsonl \
    --output_dir /workspace/rsft_output_qwen3_1_7b_k1_s43 \
    --seed 43 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 2048 \
    --bf16 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --wandb_run_name rsft_1_7b_k1_s43
