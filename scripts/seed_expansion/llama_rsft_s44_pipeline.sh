#!/bin/bash
# LLaMA-3.1-8B RSFT k=1 seed=44 full pipeline (train + eval)
# Method A: reuse s42 CED-scored data, only vary RSFT training seed
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace/freige

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export HF_HOME=/workspace/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE_MODEL=/workspace/models/meta-llama/Meta-Llama-3.1-8B
SFT_ADAPTER=/workspace/sft_output_llama_3_1_8b
RSFT_DATA=/workspace/rsft_scored_llama_3_1_8b_k1/rsft_train.jsonl
RSFT_OUTPUT=/workspace/rsft_output_llama_3_1_8b_k1_s44
EVAL_OUTPUT=/workspace/eval_results/rsft_llama_k1_s44_eval

echo "=== Step 1: RSFT Training (seed=44) ==="
echo "Start: $(date)"
python -m freige.training.rsft_trainer \
    --base_model $BASE_MODEL \
    --sft_adapter $SFT_ADAPTER \
    --rsft_data_path $RSFT_DATA \
    --output_dir $RSFT_OUTPUT \
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
echo "Step 1 done: $(date)"

echo "=== Step 2: D076 Eval ==="
echo "Start: $(date)"
python -m freige.eval.inference \
    --model_path $RSFT_OUTPUT \
    --base_model $BASE_MODEL \
    --sft_adapter $SFT_ADAPTER \
    --data_path /workspace/data/docred \
    --output_dir $EVAL_OUTPUT \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
echo "Step 2 done: $(date)"

echo "=== Pipeline Complete ==="
