#!/bin/bash
# plan_003: EDCR cross-method diagnostic table
#
# Input:  各方法的 eval 输出 (metrics.json / results.json)，路径定义在
#         freige/eval/cross_method_table.py 的 EVAL_CONFIGS 中
# Output: /workspace/eval_results/cross_method_table.json
#         + 控制台打印对比表（含 rel_f1, evi_f1, EDCR, Δ 等）
#
# 方法列表:
#   - SFT baseline (quant / noquant / no-evidence)
#   - RSFT-CED (s42/s43/s44 三 seed + r2a/r2b 变体)
#   - RSFT-flatNLI
#   - CED-reranker (N=8)
#   - DPO-CED
#   - GRPO bf16 (G=8)
#
# 缺失的 eval 结果会打印 MISSING 并跳过，不影响其他方法

set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

cd /workspace

echo "=== plan_003: EDCR Cross-Method Diagnostic ==="
echo ""

python -m freige.eval.cross_method_table

echo ""
echo "Table exported to /workspace/eval_results/cross_method_table.json"
