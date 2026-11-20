#!/usr/bin/env bash
# Executable-reliability check in the sense of Vangala, Adibifar, Gehani and Malik (2026): can the
# package install and its tests run in a clean virtual environment from the claimed dependencies alone?
# Creates a fresh venv (no system site packages), installs requirements.txt, installs the package,
# runs the tests, and writes the runtime dependency set it observed next to the claimed one.
# TORCH_INDEX=cpu (default) needs no GPU and runs the CPU tests. TORCH_INDEX=cu128 installs the CUDA
# build the results were made with, requires a working GPU, and also runs the GPU tests (set
# WAYS_DATA so they find their clip). Each index writes its own log, runtime set and lock file.
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_INDEX=${TORCH_INDEX:-cpu}
SUFFIX=$([ "$TORCH_INDEX" = cpu ] && echo "" || echo "_$TORCH_INDEX")
LOCK=provenance/requirements-lock${SUFFIX//_/-}.txt
export RUN_JSON=provenance/clean_env_run$SUFFIX.json
exec > >(sed -u -E "s|$PWD/||g; s|$HOME|~|g; s|/tmp/[^ :,]*|<tmp>|g" | tee provenance/clean_env_run$SUFFIX.log) 2>&1   # no absolute, home or temporary paths in the public log
VENV=${VENV:-$(mktemp -d)/ways-clean-venv}
PYTHON=${PYTHON:-python3}
echo "clean venv at <tmp>/${VENV##*/}, $($PYTHON --version), torch index $TORCH_INDEX"
$PYTHON -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --extra-index-url "https://download.pytorch.org/whl/$TORCH_INDEX" -r requirements.txt
"$VENV/bin/pip" install -q --no-deps -e .
"$VENV/bin/pip" check                  # the pinned set satisfies every package's declared requirements, ours included
if [ "$TORCH_INDEX" = cpu ]; then
  "$VENV/bin/python" -m pytest tests -q -m "not gpu and not slow"
else
  "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
  "$VENV/bin/python" -m pytest tests -q -rs
fi
"$VENV/bin/ways" explain --help > /dev/null
"$VENV/bin/python" - <<'PY'
import json, os, sys
import ways, ways.attribution, ways.metrics, ways.quality, ways.plots, ways.cli, ways.backends.openai_compat
# every third-party module the package, the experiments and the figure scripts import (no weights, no GPU):
# an ImportError here is a claimed-versus-working gap the tests alone would not show
import aadd.infer, aadd.v3.synth, aadd.v3.model, jiwer, librosa, openai, pandas, PIL.Image, playwright.sync_api  # noqa: F401
from transformers import AutoProcessor, MoonshineForConditionalGeneration, WhisperForConditionalGeneration  # noqa: F401
from huggingface_hub import scan_cache_dir  # noqa: F401
from ways.provenance import claimed_dependencies, runtime_dependencies
rt, cl = runtime_dependencies(), claimed_dependencies()
gap = sorted({k.lower() for k in rt} - set(cl) - {"ways"})
out = dict(python=sys.version.split()[0], n_claimed=len(cl), n_runtime=len(rt), runtime_not_claimed=gap, runtime=rt)
json.dump(out, open(os.environ["RUN_JSON"], "w"), indent=1)
print(f"claimed {len(cl)} packages, runtime {len(rt)}, multiplier {len(rt)/max(len(cl),1):.1f}x; not claimed: {len(gap)}")
PY
# the full transitive closure the clean venv resolved, installable with pip install -r <lock file>
{ echo "--extra-index-url https://download.pytorch.org/whl/$TORCH_INDEX"; "$VENV/bin/pip" freeze --exclude-editable | grep -v "^ways"; } > "$LOCK"
echo "lock file: $(grep -c "==\| @ " "$LOCK") pinned distributions in $LOCK"
echo "clean-environment run passed"
