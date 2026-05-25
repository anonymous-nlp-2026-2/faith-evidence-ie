#!/bin/bash
# 8B SFT seed=44 (canonical hyperparam, only seed+output_dir differ)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.training.sft_trainer \
    --model_name /workspace/models/Qwen/Qwen3-8B \
    --data_dir /workspace/data/docred \
    --output_dir /workspace/sft_output_qwen3_8b_s44 \
    --no-quantize \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --max_length 4096 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --bf16 \
    --seed 44 \
    --wandb_project freige-sft \
    --wandb_run_name sft-8b-s44
