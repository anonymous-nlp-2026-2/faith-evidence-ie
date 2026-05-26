#!/bin/bash
set -e
source /root/miniconda3/bin/activate
cd .

export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2

python -m freige.training.grpo_trainer \
  --model_name Qwen/Qwen3-4B \
  --sft_adapter ./sft_output \
  --data_dir data/docred \
  --stage 3 \
  --no-quantize \
  --tau_start 0.3 \
  --tau_end 0.5 \
  --output_dir ./grpo_bf16_g4 \
  --num_generations 4 \
  --reward_device cuda \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_epochs 1 \
  --learning_rate 5e-6 \
  --seed 42 \
  --kl_coef 0.0 \
  --save_steps 20 \
  --save_total_limit 2 \
  --max_steps 200 \
  --format_reward_weight 0.3 \
  --f1_reward_weight 3.0 \
  --ced_reward_weight 1.0 \
  --num_iterations 1 \
  --ced_recall_penalty \
  --wandb_run_name grpo-bf16-g4-kl0
