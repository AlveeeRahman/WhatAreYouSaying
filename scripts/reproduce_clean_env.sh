#!/usr/bin/env bash
# Executable-reliability check in the sense of Vangala, Adibifar, Gehani and Malik (2026): can the
# package install and its tests run in a clean virtual environment from the claimed dependencies alone?
# Creates a fresh venv (no system site packages), installs requirements.txt with a CPU-only torch
# index so no GPU is needed, installs the package, runs the CPU test suite, and writes the runtime
# dependency set it observed next to the claimed one.
set -euo pipefail
cd "$(dirname "$0")/.."
exec > >(sed -u -E "s|$PWD/||g; s|$HOME|~|g; s|/tmp/[^ :,]*|<tmp>|g" | tee provenance/clean_env_run.log) 2>&1   # no absolute, home or temporary paths in the public log
VENV=${VENV:-$(mktemp -d)/ways-clean-venv}
PYTHON=${PYTHON:-python3}
echo "clean venv at <tmp>/${VENV##*/}, $($PYTHON --version)"
$PYTHON -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
"$VENV/bin/pip" install -q --no-deps -e .
"$VENV/bin/pip" check                  # the pinned set satisfies every package's declared requirements, ours included
"$VENV/bin/python" -m pytest tests -q -m "not gpu and not slow"
"$VENV/bin/ways" explain --help > /dev/null
"$VENV/bin/python" - <<'PY'
import json, sys
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
json.dump(out, open("provenance/clean_env_run.json", "w"), indent=1)
print(f"claimed {len(cl)} packages, runtime {len(rt)}, multiplier {len(rt)/max(len(cl),1):.1f}x; not claimed: {len(gap)}")
PY
# the full transitive closure the clean venv resolved (CPU torch), installable with pip install -r provenance/requirements-lock.txt
{ echo "--extra-index-url https://download.pytorch.org/whl/cpu"; "$VENV/bin/pip" freeze --exclude-editable | grep -v "^ways"; } > provenance/requirements-lock.txt
echo "lock file: $(grep -c "==\| @ " provenance/requirements-lock.txt) pinned distributions in provenance/requirements-lock.txt"
echo "clean-environment run passed"
