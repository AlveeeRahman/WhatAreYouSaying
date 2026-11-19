"""Two VoxCeleb clips from the 100-clip sample through the three models: one that AADD rates clean
(no flagged frame) and one it rates degraded (every frame flagged), drawn from the saved
attributions. Usage: python experiments/vox_mixed_figures.py [data folder]"""
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from common import AADD_CSV, AUDIO_DIR, data_root

from ways.attribution import Attribution
from ways.plots import three_models_figure
from ways.quality import load_scan_csv

HERE = Path(__file__).resolve().parent
DATA = data_root(sys.argv[1] if len(sys.argv) > 1 else None, need=(AUDIO_DIR, AADD_CSV))
AUDIO = DATA / AUDIO_DIR
AADD_CSV = DATA / AADD_CSV
FIG = HERE.parent / "figures"
MODELS = ["whisper-tiny", "whisper-base", "moonshine-base"]
PICK = {"mixed": "id10274_89UnsytX6Y8_00006", "clean": "id10271_gO805KoL2RM_00006", "degraded": "id10295_K1gPy1UKmBE_00007"}
TAG = {"mixed": "partly degraded", "clean": "clean", "degraded": "degraded"}

aadd = load_scan_csv(AADD_CSV)
clips = {m: pd.read_csv(HERE / "results" / m / "clips.csv").set_index("clip") for m in MODELS}
out = {}
for tag, clip in PICK.items():
    wav, _ = sf.read(AUDIO / f"{clip}.wav", dtype="float32")
    atts = {m: Attribution.load(HERE / "results" / m / "npz" / f"{clip}.npz") for m in MODELS}
    wers = {m: float(clips[m].loc[clip, "wer"]) for m in MODELS}
    fl = float(clips[MODELS[0]].loc[clip, "frac_flagged"])
    gold = clips[MODELS[0]].loc[clip, "gold"]
    three_models_figure(atts, aadd[clip], wav, FIG / f"vox_{tag}__{clip}.png",
                        title=f"VoxCeleb, AADD rates it {TAG[tag]} ({fl:.0%} of frames flagged): {clip}\ngold: '{gold}'", wers=wers)
    out[tag] = dict(clip=clip, frac_flagged=fl, wer=wers, gold=gold, text={m: atts[m].text for m in MODELS})
    print(tag, clip, wers)
(HERE / "results/vox_mixed_figures.json").write_text(json.dumps(out, indent=1))
