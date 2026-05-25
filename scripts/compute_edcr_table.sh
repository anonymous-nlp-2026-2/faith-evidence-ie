#!/bin/bash
# Cross-method EDCR diagnostic table.
# Usage: bash scripts/compute_edcr_table.sh
#
# Reads all metrics.json from eval_results/ and generates a comparison table.

set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /workspace

python -m freige.eval.cross_method_table "$@"
