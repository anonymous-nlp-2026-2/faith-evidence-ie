#!/bin/bash
set -e

# Plan 010: NLI Model Ablation — DeBERTa-v3-large CED scoring
# Baseline: v3-base → ign_f1=0.4517
# This script: v3-large → score → filter → RSFT train → D076 eval

export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd .

SCORED_DIR=./rsft_scored_v3_large
TRAIN_DATA=$SCORED_DIR/rsft_train.jsonl
RSFT_OUTPUT=./rsft_output_nli_large
EVAL_OUTPUT=eval_results/plan010_nli_large_ablation

echo "=== Step 1: CED Scoring with v3-large ==="
echo "Start: $(date)"
python -m freige.training.rsft_score_filter \
    --input_dir ./rsft_generations \
    --output_path $SCORED_DIR \
    --nli_model_path cross-encoder/nli-deberta-v3-large \
    --nli_device cuda:0 \
    --selection_strategy top_k --top_k 1 \
    --tau 0.5 --f1_threshold 0.1
echo "Scoring done: $(date)"
echo "Training samples: $(wc -l < $TRAIN_DATA)"

echo "=== Step 2: RSFT Training ==="
echo "Start: $(date)"
python -m freige.training.rsft_trainer \
    --base_model Qwen/Qwen3-4B \
    --sft_adapter ./sft_output \
    --rsft_data_path $TRAIN_DATA \
    --output_dir $RSFT_OUTPUT \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --weighted_sampling \
    --eval_data_dir data/docred \
    --eval_max_docs 50 \
    --wandb_project freige-rsft \
    --wandb_run_name plan010-nli-large \
    --seed 42
echo "Training done: $(date)"

echo "=== Step 3: D076 Eval ==="
echo "Start: $(date)"
python -m freige.eval.inference \
    --model_path $RSFT_OUTPUT \
    --base_model Qwen/Qwen3-4B \
    --sft_adapter ./sft_output \
    --data_path data/docred \
    --output_dir $EVAL_OUTPUT \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --no-quantize \
    --seed 42
echo "Eval done: $(date)"

echo "=== ALL DONE ==="
