#!/bin/bash
# 14B SFT seed=42 (2-GPU DeepSpeed ZeRO-3, bf16 LoRA)
# VRAM: ~14GB/card (sharded params) + ~3GB (activations+LoRA) ≈ 17GB/card
# Token length analysis (Qwen3-14B tokenizer, train_annotated, 3027 docs):
#   mean=864, median=813, p95=1419, p99=1814, max=2977
#   Truncation@2048: 0.5% (14/3027) — all high-relation docs (avg 48 rels)
#   max_length=2048 is safe for this data distribution
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0,1}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

accelerate launch \
    --config_file configs/accelerate_2gpu_zero3.yaml \
    -m freige.training.sft_trainer \
    --model_name Qwen/Qwen3-14B \
    --data_dir data/docred \
    --output_dir ./sft_output_qwen3_14b \
    --deepspeed configs/deepspeed_zero3.json \
    --no-quantize \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --max_length 2048 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --bf16 \
    --seed 42 \
    --wandb_project freige-sft \
    --wandb_run_name sft-14b-s42
