#!/bin/bash
# LLaMA-3.1-8B SFT seed=43 train + eval pipeline
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTPUT_DIR=/workspace/sft_output_llama_3_1_8b_s43

echo "=== Step 1: SFT Training (seed=43) ==="
echo "Start: $(date)"
python -m freige.training.sft_trainer \
    --model_name /workspace/models/meta-llama/Meta-Llama-3.1-8B \
    --data_dir /workspace/data/docred \
    --output_dir $OUTPUT_DIR \
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
    --seed 43 \
    --wandb_project freige-sft \
    --wandb_run_name sft-llama-s43
echo "Step 1 done: $(date)"

echo "=== Step 2: D076 Eval ==="
echo "Start: $(date)"
python -m freige.eval.inference \
    --model_path $OUTPUT_DIR \
    --base_model /workspace/models/meta-llama/Meta-Llama-3.1-8B \
    --data_path /workspace/data/docred \
    --output_dir /workspace/eval_results/sft_llama_s43_eval \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
echo "Step 2 done: $(date)"

echo "=== Pipeline Complete ==="
