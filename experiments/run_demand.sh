#!/usr/bin/env bash
# VoiceBank-DEMAND: prepare once, then the three models over 100 clips x 3 conditions, 30 minutes at most per model.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
MAX_MIN=${MAX_MIN:-30}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} PYTHONUNBUFFERED=1
[ -f experiments/results/demand/manifest.csv ] || $PY experiments/demand.py prepare
for spec in "whisper-base hf-whisper openai/whisper-base" \
            "moonshine-base hf-moonshine UsefulSensors/moonshine-base" \
            "whisper-tiny hf-whisper openai/whisper-tiny"; do
  set -- $spec
  echo "=== demand $1 $(date -u +%H:%M:%S)"
  $PY experiments/demand.py run "$1" "$2" "$3" "$MAX_MIN" 2>&1 | grep --line-buffered -v -E "torchao|Warning|logits processor|Loading weights"
done
$PY experiments/demand.py summarise 2>&1 | grep -v -E "torchao|Warning"
echo "=== demand all done $(date -u +%H:%M:%S)"
