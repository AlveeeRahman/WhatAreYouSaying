"""The two things every experiment script shares: where the pipeline repository is, and which exact
Hugging Face snapshot of each model the results were produced with.

Pipeline repository (the pipeline repository, the earlier study): it holds the VoxCeleb
audio, the saved test slice, the SLUE transcripts and the AADD scan. It is found, in this order, from
the command-line argument a script documents, the WAYS_PIPELINE environment variable, or a checkout
next to this repository (../ways-data). If none exists the script stops
with a message saying so.

Model revisions: the snapshot commits recorded in experiments/results/<model>/provenance.json
(`models` field). Passing them as `revision=` makes a re-run load the same weights even after the
Hub repository moves on.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PIPE = REPO.parent / "pipeline"

REVISIONS = {
    "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "openai/whisper-base": "e37978b90ca9030d5170a5c07aadb050351a65bb",
    "UsefulSensors/moonshine-base": "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf",
}


def pipeline_repo(arg: str | None = None, need: tuple[str, ...] = (".",)) -> Path:
    """Resolve the pipeline repository and check that the parts a script reads are there."""
    p = Path(arg or os.environ.get("WAYS_PIPELINE") or DEFAULT_PIPE).expanduser()
    missing = [n for n in need if not (p / n).exists()]
    if missing:
        sys.exit(f"pipeline repository not found or incomplete at {p} (missing: {', '.join(missing)}).\n"
                 "Prepare the data folder (the earlier study, with its VoxCeleb audio) and pass "
                 "its path as the argument, or set WAYS_PIPELINE=/path/to/it.")
    return p
