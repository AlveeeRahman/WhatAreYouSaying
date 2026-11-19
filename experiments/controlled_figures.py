"""Figures for the controlled-degradation experiment: one clip clean and degraded through the three
models with the degraded window marked, and the summary bars per degradation class."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from ways.attribution import Attribution
from ways.plots import BG, BORDER, TXT, TXT2, CAT, DPI, _label, three_models_figure
from ways.quality import CLASSES as AADD_CLASSES

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache/controlled"
OUT = HERE / "results/controlled"
FIG = HERE.parent / "figures"
MODELS = ["whisper-tiny", "whisper-base", "moonshine-base"]
DEG = ["noise", "reverb", "clipping", "lowpass", "hum", "clicks"]
man = pd.read_csv(OUT / "manifest.csv").set_index("id")
clips = pd.read_csv(OUT / "clips.csv")


def example(cls="noise"):
    """The clip of this class where whisper-base lost the most words to the degradation."""
    c = clips[(clips.model == "whisper-base") & (clips.degradation == cls)]
    cid = c.sort_values("wer_deg", ascending=False).id.iloc[0]
    a, b = man.loc[cid, "win_start"], man.loc[cid, "win_end"]
    for cond in ("clean", "deg"):
        wav, _ = sf.read(CACHE / f"{cid}_{cond}.wav", dtype="float32")
        probs = np.load(CACHE / f"{cid}_aadd.npy") if cond == "deg" else np.zeros((len(wav) // 320, len(AADD_CLASSES)), np.float32)
        atts = {m: Attribution.load(OUT / m / "npz" / f"{cid}_{cond}.npz") for m in MODELS}
        wers = {m: float(clips[(clips.model == m) & (clips.id == cid)][f"wer_{'deg' if cond == 'deg' else 'clean'}"].iloc[0]) for m in MODELS}
        what = f"{cls} inserted at {a:.1f}-{b:.1f} s (dashed)" if cond == "deg" else "clean original"
        three_models_figure(atts, probs, wav, FIG / f"controlled_{cls}_{cond}__{cid}.png", wers=wers, window=(a, b),
                            title=f"LibriSpeech {cid}, {what}\ngold: '{man.loc[cid, 'text']}'")
    return cid, (a, b)


def summary():
    S = json.load(open(OUT / "summary.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), gridspec_kw=dict(left=0.06, right=0.98, top=0.8, bottom=0.2, wspace=0.34))
    w = 0.8 / len(MODELS)
    x = np.arange(len(DEG))
    for i, m in enumerate(MODELS):
        bc = S["models"][m]["by_class"]
        ax = axes[0]
        ax.bar(x + (i - 1) * w, [bc[c]["wer_deg"] - bc[c]["wer_clean"] for c in DEG], width=w, color=CAT[i], label=m)
        ax = axes[1]
        ax.bar(x + (i - 1) * w, [bc[c]["mass_in_deg"] - bc[c]["mass_in_clean"] for c in DEG], width=w, color=CAT[i])
        ax = axes[2]
        ax.bar(x + (i - 1) * w, [bc[c]["ds_in"] for c in DEG], width=w, color=CAT[i])
    for ax, lab, ttl in zip(axes, ["WER rise, degraded minus clean", "change in mass inside the window", "surprisal rise (nats), in-window tokens"],
                            ["(a) Words lost to the degradation", "(b) Do in-window tokens still use the window", "(c) How much more surprised the model is"]):
        ax.axhline(0, color=TXT2, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(DEG, rotation=20); ax.set_ylabel(lab); _label(ax, ttl)
        for s in ax.spines.values():
            s.set_color(BORDER)
    axes[0].legend(frameon=False)
    fig.suptitle("60 LibriSpeech clips, one 2 s degradation each, clean versus degraded, per model", x=0.06, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.97)
    fig.savefig(FIG / "controlled_summary.png", dpi=DPI); fig.savefig(FIG / "controlled_summary.svg"); plt.close(fig)


if __name__ == "__main__":
    picks = {cls: example(cls) for cls in ("noise", "reverb")}
    summary()
    (OUT / "figure_picks.json").write_text(json.dumps(picks, indent=1, default=str))
    print(picks)
