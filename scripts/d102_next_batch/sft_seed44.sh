#!/bin/bash
set -e
source /root/miniconda3/bin/activate
cd .

export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

python -m freige.training.sft_trainer \
  --model_name Qwen/Qwen3-4B \
  --data_dir data/docred \
  --output_dir ./sft_output_seed44 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --max_length 4096 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.05 \
  --bf16 \
  --seed 44 \
  --wandb_project freige-sft \
  --wandb_run_name sft-4b-seed44
