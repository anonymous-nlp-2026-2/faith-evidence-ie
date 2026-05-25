#!/bin/bash
# Test set eval: Qwen3-8B SFT no-evidence (dev ign_f1=0.5079)
# Output: codalab_submission.json for DocRED leaderboard
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

export GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

cd /workspace/freige

python -m freige.eval.inference \
    --model_path /workspace/models/Qwen/Qwen3-8B \
    --base_model /workspace/models/Qwen/Qwen3-8B \
    --sft_adapter /workspace/sft_output_qwen3_8b_no_evidence \
    --data_path /workspace/data/docred \
    --split test \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --seed 42 \
    --no_evidence \
    --no-quantize \
    --output_dir /workspace/eval_results/test_8b_sft_no_evidence_best

echo ""
echo "Done. Submit codalab_submission.json to DocRED CodaLab."
