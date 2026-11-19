"""The per-word figure on one alternating VoiceBank-DEMAND clip through the three models, from the saved
attributions: AADD rows, the true noisy segments as dashed windows, one box per word. The clip is
p257_003 (bus noise at 7.5 dB, pattern noisy, clean, noisy, clean, clean, clean, noisy).
Usage: python experiments/demand_figures.py [clip id]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ways.attribution import Attribution
from ways.plots import three_models_figure

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache/demand"
RES = HERE / "results/demand"
FIG = HERE.parent / "figures"
MODELS = ["whisper-tiny", "whisper-base", "moonshine-base"]
CLIP = sys.argv[1] if len(sys.argv) > 1 else "p257_003"

man = pd.read_csv(RES / "manifest.csv").set_index("id")
segs = json.loads(man.loc[CLIP, "segments"])
wins = [(a, b) for a, b, s in segs if s]
pattern = ", ".join("noisy" if s else "clean" for _, _, s in segs)
out = {}
for cond in ("alt", "noisy"):
    wav, _ = sf.read(CACHE / f"{CLIP}_{cond}.wav", dtype="float32")
    probs = np.load(CACHE / f"{CLIP}_aadd_{cond}.npy")
    atts, wers = {}, {}
    for m in MODELS:
        p = RES / m / "npz" / f"{CLIP}_{cond}.npz"
        if p.exists():
            atts[m] = Attribution.load(p)
            rows = pd.DataFrame([json.loads(l) for l in open(RES / m / "rows.jsonl")])
            wers[m] = float(rows[(rows.id == CLIP) & (rows.cond == cond)].wer.iloc[-1])
    if not atts:
        continue
    what = f"real {man.loc[CLIP, 'noise']} noise at {man.loc[CLIP, 'snr_db']:g} dB switched on and off ({pattern})" if cond == "alt" else \
           f"real {man.loc[CLIP, 'noise']} noise at {man.loc[CLIP, 'snr_db']:g} dB over the whole clip"
    title = f"VoiceBank-DEMAND {CLIP}: {what}\nsentence: '{man.loc[CLIP, 'text']}'"
    three_models_figure(atts, probs, wav, FIG / f"demand_{cond}__{CLIP}.png", title=title, wers=wers, window=wins if cond == "alt" else None)
    out[cond] = dict(clip=CLIP, segments=segs, wer=wers, text={m: a.text for m, a in atts.items()})
    print(cond, CLIP, wers)
(RES / "example_figures.json").write_text(json.dumps(out, indent=1))
