#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}                       # the interpreter of the venv from the README
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} PYTHONUNBUFFERED=1
experiments/run_detectors.sh
echo "=== backbones $(date -u +%H:%M:%S)"
$PY experiments/detector_backbones.py 2>&1 | grep --line-buffered -v -E "torchao|Warning|Loading weights"
echo "=== backbones done $(date -u +%H:%M:%S)"
