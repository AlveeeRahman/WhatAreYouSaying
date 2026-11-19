"""The example clip (id10288_FIxC_PNJNkc_00003, the same one used throughout the earlier figures)
through the three models: saves the attributions and draws the method figure (whisper-base), the
per-model clip figures and the three-model comparison. Usage: python experiments/example_clip.py [pipeline repo]"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from ways import explain, load_backend
from ways.attribution import Attribution
from ways.metrics import clip_summary, token_summary, wer
from ways.plots import clip_figure, method_figure, three_models_figure
from ways.quality import load_scan_csv
from common import REVISIONS, pipeline_repo

CLIP = "id10288_FIxC_PNJNkc_00003"
HERE = Path(__file__).resolve().parent
PIPE = pipeline_repo(sys.argv[1] if len(sys.argv) > 1 else None, need=("audio", "results"))
AUDIO = PIPE / "audio"
GOLD = PIPE / "transcripts.tsv"
AADD_CSV = PIPE / "aadd_frames_20ms.csv"
FIG = HERE.parent / "figures"; FIG.mkdir(exist_ok=True)
OUT = HERE / "results/example"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = [("whisper-tiny", "hf-whisper", "openai/whisper-tiny"), ("whisper-base", "hf-whisper", "openai/whisper-base"),
          ("moonshine-base", "hf-moonshine", "UsefulSensors/moonshine-base")]

wav, sr = sf.read(AUDIO / f"{CLIP}.wav", dtype="float32")
import pandas as pd
gold = dict(pd.read_csv(GOLD, sep="\t", usecols=["id", "normalized_text"], keep_default_na=False).itertuples(index=False))
probs = load_scan_csv(AADD_CSV)[CLIP]
atts, wers, summ = {}, {}, {}
for name, kind, mid in MODELS:
    f = OUT / f"{name}.npz"
    if f.exists():
        att = Attribution.load(f)
    else:
        att = explain(load_backend(kind, mid, revision=REVISIONS.get(mid)), wav, batch=64)
        att.save(f)
    atts[name] = att
    w = wer(gold[CLIP], att.text); wers[name] = w["wer"]
    rows = token_summary(att, probs)
    summ[name] = dict(clip_summary(att, rows, probs), wer=w["wer"], text=att.text)
    clip_figure(att, probs, wav, FIG / f"{name}__{CLIP}.png", title=f"{CLIP}\n{name}  ·  WER {w['wer']:.0%}  ·  {att.delta.shape[0]} tokens")
    print(name, f"WER {w['wer']:.2f}", att.text)
k, patch = method_figure(atts["whisper-base"], probs, wav, FIG / "method_whisper_base.png",
                         title=f"Representation erasure on audio: whisper-base, clip {CLIP}")
summ["method_figure"] = dict(model="whisper-base", token=int(k), piece=atts["whisper-base"].pieces[k], patch=[int(v) for v in patch])
three_models_figure(atts, probs, wav, FIG / f"three_models__{CLIP}.png", title=f"Same clip, three models: {CLIP}  ·  gold: '{gold[CLIP]}'", wers=wers)
(OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
print(json.dumps(summ["method_figure"]))
