"""The latent view over the 100 VoxCeleb clips (AADD flags) and the 60 controlled degraded clips (known
window), for the three speech-to-text models. Per clip and layer: spill-over inside and outside the
degraded stretch, CAM focus, correlation of the CAM with loudness. Resumable; writes
experiments/results/latent/<model>/{vox,controlled}.jsonl, summary.json and summary.md.

Usage: python experiments/latent_run.py <name> <backend kind> <model id> [data folder]
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from common import AADD_CSV, AUDIO_DIR, REVISIONS, data_root

from ways import load_backend
from ways.latent import latent_view
from ways.quality import load_scan_csv

name, kind, model_id = sys.argv[1:4]
HERE = Path(__file__).resolve().parent
DATA = data_root(sys.argv[4] if len(sys.argv) > 4 else None, need=(AUDIO_DIR, AADD_CSV))
AUDIO = DATA / AUDIO_DIR
AADD_CSV = DATA / AADD_CSV
CACHE = HERE / "cache/controlled"
OUT = HERE / "results/latent" / name; OUT.mkdir(parents=True, exist_ok=True)
be = load_backend(kind, model_id, revision=REVISIONS.get(model_id))
sample = pd.read_csv(HERE / "sample100.csv")
man = pd.read_csv(HERE / "results/controlled/manifest.csv")
aadd = load_scan_csv(AADD_CSV)


def run(tag, items):
    f = OUT / f"{tag}.jsonl"
    done = {json.loads(l)["clip"] for l in open(f)} if f.exists() else set()
    for clip, wav, probs, window in items:
        if clip in done:
            continue
        t0 = time.time()
        r = latent_view(be, wav, probs=probs, window=window)
        row = dict(clip=clip, **r.to_dict(), seconds=time.time() - t0)
        with open(f, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{name} {tag} {clip} layers {r.layers} share {r.degraded.mean():.2f} {time.time() - t0:.1f}s", flush=True)


def vox_items():
    for clip in sample.file:
        wav, _ = sf.read(AUDIO / f"{clip}.wav", dtype="float32")
        yield clip, wav, aadd.get(clip), None


def controlled_items():
    for r in man.itertuples():
        wav, _ = sf.read(CACHE / f"{r.id}_deg.wav", dtype="float32")
        yield r.id, wav, None, (r.win_start, r.win_end)


run("vox", vox_items())
run("controlled", controlled_items())
from ways.provenance import write as write_manifest

write_manifest(OUT / "provenance.json", inputs={"sample100.csv": str(HERE / "sample100.csv"), "aadd_frames_csv": str(AADD_CSV),
                                                "controlled_manifest.csv": str(HERE / "results/controlled/manifest.csv")},
               seeds={"sample": 8001, "bootstrap": 8001}, models={model_id: REVISIONS.get(model_id)},
               extra=dict(backend=be.describe(), passes=["vox", "controlled"]))

# summary per layer, bootstrap over clips
from ways.metrics import bootstrap_ci

S = {}
for tag in ("vox", "controlled"):
    rows = [json.loads(l) for l in open(OUT / f"{tag}.jsonl")]
    if tag == "vox":
        rows = [r for r in rows if 0.05 <= r["degraded_share"] <= 0.95]   # informative clips only, same rule as compare.py
    L = rows[0]["layers"]
    S[tag] = dict(n_clips=len(rows), layers=[])
    for l in range(L):
        get = lambda k, l=l, rows=rows: [r["summary"][l][k] for r in rows]   # bound per layer
        S[tag]["layers"].append(dict(layer=l, shift_out=bootstrap_ci(get("shift_out")), shift_in=bootstrap_ci(get("shift_in")),
                                     cam_focus=bootstrap_ci(get("cam_focus")), rho_loud=bootstrap_ci(get("rho_loud"))))
(OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))
ci = lambda d: f"{d['mean']:.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"
lines = []
for tag, title in (("vox", "100 VoxCeleb clips, AADD flags (clips with 5 to 95 % flagged)"), ("controlled", "60 controlled clips, known 2 s window")):
    lines += [f"**{name}, {title}: {S[tag]['n_clips']} clips**", "", "| layer | spill-over outside | shift inside | CAM focus on degraded | rho(CAM, loudness) |", "|---|---|---|---|---|"]
    for d in S[tag]["layers"]:
        lines.append(f"| {d['layer']} | {ci(d['shift_out'])} | {ci(d['shift_in'])} | {ci(d['cam_focus'])} | {ci(d['rho_loud'])} |")
    lines.append("")
(OUT / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
