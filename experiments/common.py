"""The two things every experiment script shares: where the VoxCeleb data folder is, and which exact
Hugging Face snapshot of each model the results were produced with.

Data folder: a plain directory with the VoxCeleb material the 100-clip experiments read. It is not
part of this repository (the audio is not redistributed). Layout:

    audio/<clip id>.wav       16 kHz mono clips, named by their VoxCeleb id
    test.csv                  the fixed test slice (columns: file, label, original_index, duration)
    development.csv           the development slice (column: file), only used to assert disjointness
    heldout.csv               the held-out slice (column: file), same use; its audio is never read
    transcripts.tsv           human transcripts (columns: id, normalized_text)
    aadd_frames_20ms.csv      AADD v3.1 frame probabilities for every clip (the AADD CLI scan output)

It is found, in this order, from the command-line argument a script documents, the WAYS_DATA
environment variable, or a folder named ways-data next to this repository. If none exists the
script stops with a message saying so.

Model revisions: the Hugging Face snapshot commits the results were produced with. Passing them as
`revision=` makes a re-run load the same weights even after the Hub repository moves on.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO.parent / "ways-data"
AUDIO_DIR, TEST_CSV, DEV_CSV, HELDOUT_CSV, TRANSCRIPTS_TSV, AADD_CSV = (
    "audio", "test.csv", "development.csv", "heldout.csv", "transcripts.tsv", "aadd_frames_20ms.csv")

REVISIONS = {
    "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "openai/whisper-base": "e37978b90ca9030d5170a5c07aadb050351a65bb",
    "UsefulSensors/moonshine-base": "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf",
}


def data_root(arg: str | None = None, need: tuple[str, ...] = (AUDIO_DIR,)) -> Path:
    """Resolve the data folder and check that the parts a script reads are there."""
    p = Path(arg or os.environ.get("WAYS_DATA") or DEFAULT_DATA).expanduser()
    missing = [n for n in need if not (p / n).exists()]
    if missing:
        sys.exit(f"data folder not found or incomplete at {p} (missing: {', '.join(missing)}).\n"
                 "Point WAYS at a folder with the layout described in experiments/common.py: pass its path as the "
                 "argument, or set WAYS_DATA=/path/to/it.")
    return p


def resume_rows(rows_f: Path, has_outputs) -> list[dict]:
    """The rows of a finished-work file whose saved outputs still exist.

    A row counts as done only when `has_outputs(row)` is true. Rows whose outputs are missing (a
    fresh clone has the tracked rows but not the ignored npz maps) are dropped from the file, so the
    run that follows recomputes them and writes each row once.
    """
    if not rows_f.exists():
        return []
    rows = [json.loads(line) for line in open(rows_f)]
    kept = [r for r in rows if has_outputs(r)]
    if len(kept) != len(rows):
        rows_f.write_text("".join(json.dumps(r) + "\n" for r in kept))
    return kept
