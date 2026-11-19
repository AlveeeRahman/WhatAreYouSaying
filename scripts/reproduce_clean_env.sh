#!/usr/bin/env bash
# Executable-reliability check in the sense of Vangala, Adibifar, Gehani and Malik (2026): can the
# package install and its tests run in a clean virtual environment from the claimed dependencies alone?
# Creates a fresh venv (no system site packages), installs requirements.txt with a CPU-only torch
# index so no GPU is needed, installs the package, runs the CPU test suite, and writes the runtime
# dependency set it observed next to the claimed one.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-$(mktemp -d)/ways-clean-venv}
PYTHON=${PYTHON:-python3}
echo "clean venv at $VENV"
$PYTHON -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
"$VENV/bin/pip" install -q --no-deps -e .
"$VENV/bin/python" -m pytest tests -q -m "not gpu and not slow"
"$VENV/bin/python" - <<'PY'
import json, sys
import ways, ways.attribution, ways.metrics, ways.quality, ways.plots, ways.cli, ways.backends.openai_compat
from ways.provenance import claimed_dependencies, runtime_dependencies
rt, cl = runtime_dependencies(), claimed_dependencies()
gap = sorted({k.lower() for k in rt} - set(cl) - {"ways"})
out = dict(python=sys.version.split()[0], n_claimed=len(cl), n_runtime=len(rt), runtime_not_claimed=gap, runtime=rt)
json.dump(out, open("provenance/clean_env_run.json", "w"), indent=1)
print(f"claimed {len(cl)} packages, runtime {len(rt)}, multiplier {len(rt)/max(len(cl),1):.1f}x; not claimed: {len(gap)}")
PY
echo "clean-environment run passed"
