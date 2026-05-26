#!/bin/bash
# Step 3: RSFT training for 14B (Qwen3-14B + SFT merged → RSFT LoRA)
# Prerequisite: SFT adapter merged into base model (ZeRO-3 incompatible with runtime merge)
# Uses: accelerate + DeepSpeed ZeRO-3 on 2 GPUs
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd .

export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-2,3}}
export HF_HOME=./.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MERGED_MODEL="Qwen/Qwen3-14B-sft-merged"
RSFT_DATA="./rsft_scored_14b/rsft_train.jsonl"
OUTPUT_DIR="./rsft_output_14b"

# --- Pre-flight checks ---
if [ ! -d "$MERGED_MODEL" ]; then
    echo "ERROR: Merged SFT model not found at $MERGED_MODEL"
    echo "Run merge first:"
    echo "  CUDA_VISIBLE_DEVICES=0 python scripts/merge_sft_adapter_14b.py"
    exit 1
fi

if [ ! -f "$RSFT_DATA" ]; then
    echo "ERROR: RSFT training data not found: $RSFT_DATA"
    echo "Ensure step2_ced_score_14b.sh has completed."
    exit 1
fi

echo "=== Step 3: RSFT Training (14B) ==="
echo "Base (merged): $MERGED_MODEL"
echo "Data:          $RSFT_DATA"
echo "Output:        $OUTPUT_DIR"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"

accelerate launch \
  --config_file configs/accelerate_2gpu_zero3.yaml \
  -m freige.training.rsft_trainer \
  --base_model "$MERGED_MODEL" \
  --sft_adapter "" \
  --rsft_data_path "$RSFT_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed configs/deepspeed_zero3_14b.json \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
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
  --wandb_project freige-rsft \
  --wandb_run_name rsft-14b-plan017

echo "=== RSFT Training Complete ==="
