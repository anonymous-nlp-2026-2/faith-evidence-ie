#!/bin/bash
# Random RSFT k=1 baseline (Claim 3 ablation: random vs CED scoring)
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export CUDA_VISIBLE_DEVICES=${GPU:-0}  # removed: tmux wrapper sets this

RANDOM_DATA=/workspace/rsft_random_k1/rsft_train.jsonl

python -m freige.training.rsft_trainer \
  --base_model /workspace/models/Qwen3-4B \
  --sft_adapter /workspace/sft_output \
  --rsft_data_path "$RANDOM_DATA" \
  --output_dir /workspace/rsft_output_random_k1 \
  --learning_rate 2e-5 \
  --num_epochs 3 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --warmup_steps 100 \
  --save_steps 50 \
  --save_total_limit 2 \
  --max_length 2048 \
  --seed 42 \
  --bf16 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --wandb_project freige-rsft \
  --wandb_run_name rsft-random-k1
