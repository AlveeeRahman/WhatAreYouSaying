#!/usr/bin/env bash
# The latent view for the three models over the 100 VoxCeleb clips and the 60 controlled clips.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} PYTHONUNBUFFERED=1
for spec in "whisper-tiny hf-whisper openai/whisper-tiny" "whisper-base hf-whisper openai/whisper-base" "moonshine-base hf-moonshine UsefulSensors/moonshine-base"; do
  set -- $spec
  echo "=== latent $1 $(date -u +%H:%M:%S)"
  $PY experiments/latent_run.py "$1" "$2" "$3" 2>&1 | grep --line-buffered -v -E "torchao|Warning|Loading weights|logits processor"
done
echo "=== latent all done $(date -u +%H:%M:%S)"
