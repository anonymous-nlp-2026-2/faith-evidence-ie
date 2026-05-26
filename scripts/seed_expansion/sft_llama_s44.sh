#!/bin/bash
# LLaMA-3.1-8B SFT seed=44 (canonical hyperparam, only seed+output_dir differ)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m freige.training.sft_trainer \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --data_dir data/docred \
    --output_dir ./sft_output_llama_3_1_8b_s44 \
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
    --wandb_run_name sft-llama-s44
