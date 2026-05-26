#!/bin/bash
# Test set eval: Qwen3-8B RSFT k=1 s43 (best seed, dev ign_f1=0.5471)
# Output: codalab_submission.json for DocRED leaderboard
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

export GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

cd .

python -m freige.eval.inference \
    --model_path ./rsft_output_qwen3_8b_k1_s43 \
    --base_model Qwen/Qwen3-8B \
    --sft_adapter ./sft_output_qwen3_8b_bf16 \
    --data_path data/docred \
    --split test \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --seed 42 \
    --no-quantize \
    --output_dir eval_results/test_8b_rsft_s43_best

echo ""
echo "Done. Submit codalab_submission.json to DocRED CodaLab."
