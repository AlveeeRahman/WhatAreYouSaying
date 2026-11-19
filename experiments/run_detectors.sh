#!/usr/bin/env bash
# Train the two extra detectors one after another on LibriSpeech dev-clean.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}                       # the interpreter of the venv from the README
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} PYTHONUNBUFFERED=1
LS=experiments/cache/librispeech/LibriSpeech/dev-clean
[ -d "$LS" ] || { echo "LibriSpeech dev-clean not found at $LS: download https://www.openslr.org/resources/12/dev-clean.tar.gz and extract it into experiments/cache/librispeech/" >&2; exit 1; }
echo "=== dev-clean ready $(date -u +%H:%M:%S)"
for n in whisper-base moonshine-base; do
  echo "=== train $n $(date -u +%H:%M:%S)"
  $PY experiments/train_detectors.py "$n" --steps 4000 --bs 12 2>&1 | grep --line-buffered -v -E "torchao|Warning|Loading weights"
done
echo "=== detectors done $(date -u +%H:%M:%S)"
