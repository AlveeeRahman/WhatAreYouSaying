#!/usr/bin/env bash
# The three small speech-to-text models over the 100-clip sample, one after another on one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}                       # the interpreter of the venv from the README
# offline by default, so the cached pinned snapshots are used; on a new machine run once with
# HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 to download them. The data folder: WAYS_DATA.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
mkdir -p experiments/results
for spec in "moonshine-base hf-moonshine UsefulSensors/moonshine-base" \
            "whisper-base hf-whisper openai/whisper-base" \
            "whisper-tiny hf-whisper openai/whisper-tiny"; do
  set -- $spec
  echo "=== $1 $(date -u +%H:%M:%S)"
  $PY experiments/run_models.py "$1" "$2" "$3" 2>&1 | grep -v -E "torchao|Warning|logits processor|Loading weights"
done
echo "=== all done $(date -u +%H:%M:%S)"
