#!/usr/bin/env bash
# The three models over the 60 controlled clips, clean and degraded, one after another on one GPU.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for spec in "moonshine-base hf-moonshine UsefulSensors/moonshine-base" \
            "whisper-base hf-whisper openai/whisper-base" \
            "whisper-tiny hf-whisper openai/whisper-tiny"; do
  set -- $spec
  echo "=== controlled $1 $(date -u +%H:%M:%S)"
  $PY experiments/controlled.py run "$1" "$2" "$3" 2>&1 | grep --line-buffered -v -E "torchao|Warning|logits processor|Loading weights"
done
$PY experiments/controlled.py summarise 2>&1 | grep -v -E "torchao|Warning"
echo "=== controlled all done $(date -u +%H:%M:%S)"
