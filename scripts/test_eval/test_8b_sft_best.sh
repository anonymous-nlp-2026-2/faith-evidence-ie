#!/bin/bash
# Test set eval: Qwen3-8B SFT s43 (best seed, dev ign_f1=0.5324)
# Output: codalab_submission.json for DocRED leaderboard
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

export GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

cd /workspace/freige

python -m freige.eval.inference \
    --model_path /workspace/models/Qwen/Qwen3-8B \
    --base_model /workspace/models/Qwen/Qwen3-8B \
    --sft_adapter /workspace/sft_output_qwen3_8b_s43 \
    --data_path /workspace/data/docred \
    --split test \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --seed 42 \
    --no-quantize \
    --output_dir /workspace/eval_results/test_8b_sft_s43_best

echo ""
echo "Done. Submit codalab_submission.json to DocRED CodaLab."
