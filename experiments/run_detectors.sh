#!/usr/bin/env bash
# Wait for LibriSpeech dev-clean, then train the two extra detectors one after another.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q DONE experiments/cache/librispeech/download_dev.log 2>/dev/null; do sleep 20; done
echo "=== dev-clean ready $(date -u +%H:%M:%S)"
for n in whisper-base moonshine-base; do
  echo "=== train $n $(date -u +%H:%M:%S)"
  $PY experiments/train_detectors.py "$n" --steps 4000 --bs 12 2>&1 | grep --line-buffered -v -E "torchao|Warning|Loading weights"
done
echo "=== detectors done $(date -u +%H:%M:%S)"
