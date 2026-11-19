"""The same noise-focus question under three detectors whose backbones differ:
  whisper-small   AADD v3.1 as shipped (gertelrina/aadd, 88.6 M parameters)
  whisper-base    AADD's own frame model on the whisper-base encoder, trained here (experiments/train_detectors.py)
  moonshine-base  the same head on the Moonshine encoder, trained here
Each detector scores (a) the 60 controlled clips, where the damaged window is known, so its hit rate inside and
outside the window is measured against ground truth, and (b) the 100 VoxCeleb clips, where its flags feed the
noise-focus ratio of each speech-to-text model. The attribution maps never change; only the flags do.
Writes experiments/results/detector_backbones.{json,md} and figures/detector_backbones.png."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_detectors import frame_probs, load_detector  # noqa: E402

from ways.metrics import bootstrap_ci, paired_bootstrap  # noqa: E402
from ways.quality import CLASSES as AADD_CLASSES  # noqa: E402

PIPE = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parents[1] / "pipeline")
AUDIO = PIPE / "audio"
CACHE = HERE / "cache/controlled"
RES = HERE / "results"
FIG = HERE.parent / "figures"
ASR = ["whisper-tiny", "whisper-base", "moonshine-base"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample = pd.read_csv(HERE / "sample100.csv")
man = pd.read_csv(RES / "controlled/manifest.csv")


def detectors():
    from aadd.infer import Detector
    v3 = Detector(model="v3", resolution=0.02)
    yield "whisper-small (AADD v3.1)", lambda w: np.asarray(v3.predict(w, 16000).frame_probs, np.float32), dict(n_params=int(sum(p.numel() for p in v3.model.parameters())), trained_by="AADD")
    for name in ("whisper-base", "moonshine-base"):
        m, ck = load_detector(name, DEV)
        yield f"{name} (trained here)", (lambda w, m=m: frame_probs(m, w, DEV)), dict(n_params=int(sum(p.numel() for p in m.parameters())), trained_by="this repository", val_auc=ck["meta"]["val_auc"], steps=ck["meta"]["step"])


S = dict(detectors={}, controlled={}, focus={})
(RES / "aadd").mkdir(exist_ok=True)
for name, fn, info in detectors():
    key = name.split(" ")[0]
    S["detectors"][name] = info
    # (a) controlled set: ground-truth window
    rows = []
    for r in man.itertuples():
        f = RES / "aadd" / f"det_{key}" / f"{r.id}.npy"; f.parent.mkdir(exist_ok=True)
        if f.exists():
            P = np.load(f)
        else:
            w, _ = sf.read(CACHE / f"{r.id}_deg.wav", dtype="float32"); P = fn(w); np.save(f, P)
        t = np.arange(len(P)) * 0.02; inside = (t >= r.win_start) & (t < r.win_end); j = AADD_CLASSES.index(r.degradation)
        rows.append(dict(degradation=r.degradation, flagged_inside=float((P[inside] >= 0.5).any(1).mean()),
                         flagged_outside=float((P[~inside] >= 0.5).any(1).mean()), class_inside=float((P[inside, j] >= 0.5).mean())))
    d = pd.DataFrame(rows)
    S["controlled"][name] = dict(overall=d[["flagged_inside", "flagged_outside", "class_inside"]].mean().round(3).to_dict(),
                                 by_class=d.groupby("degradation")[["flagged_inside", "flagged_outside", "class_inside"]].mean().round(2).to_dict("index"))
    # (b) VoxCeleb: probabilities for the noise-focus ratio
    for clip in sample.file:
        f = RES / "aadd" / f"det_{key}" / f"{clip}.npy"
        if not f.exists():
            w, _ = sf.read(AUDIO / f"{clip}.wav", dtype="float32"); np.save(f, fn(w))
    print(name, S["controlled"][name]["overall"], flush=True)

# noise-focus ratio of every speech-to-text model under every detector, reusing compare.py's function
import compare  # noqa: E402  (loads the clips and tokens tables)
compare.DETS.update({f"det_{n.split(' ')[0]}": 0.02 for n in S["detectors"]})
for name in S["detectors"]:
    det = f"det_{name.split(' ')[0]}"
    S["focus"][name] = {m: bootstrap_ci(compare.focus_under(det, m)) for m in ASR}
    fb, fm = compare.focus_under(det, "whisper-base"), compare.focus_under(det, "moonshine-base")
    S["focus"][name]["paired whisper-base - moonshine-base"] = paired_bootstrap(fb, fm)
(RES / "detector_backbones.json").write_text(json.dumps(S, indent=1, default=float))

ci = lambda d, f="{:.2f}": f"{f.format(d['mean'])} [{f.format(d['ci'][0])}, {f.format(d['ci'][1])}]"
L = ["| detector backbone | parameters | trained by | flags inside the damaged 2 s | flags outside | right class inside |", "|---|---|---|---|---|---|"]
for name, info in S["detectors"].items():
    o = S["controlled"][name]["overall"]
    L.append(f"| {name} | {info['n_params'] / 1e6:.1f} M | {info['trained_by']} | {o['flagged_inside']:.2f} | {o['flagged_outside']:.2f} | {o['class_inside']:.2f} |")
L += ["", "| detector backbone | " + " | ".join(ASR) + " | paired whisper-base − moonshine-base |", "|---|" + "---|" * (len(ASR) + 1)]
for name, d in S["focus"].items():
    L.append(f"| {name} | " + " | ".join(ci(d[m]) + f" (n={d[m]['n']})" for m in ASR) + f" | {ci(d['paired whisper-base - moonshine-base'], '{:+.3f}')} |")
(RES / "detector_backbones.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ways.plots import BORDER, CAT, TXT, TXT2, DPI, _label
names = list(S["detectors"])
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(left=0.07, right=0.98, top=0.78, bottom=0.2, wspace=0.3))
ax = axes[0]; x = np.arange(len(names)); w = 0.38
ax.bar(x - w / 2, [S["controlled"][n]["overall"]["flagged_inside"] for n in names], width=w, color=CAT[0], label="inside the damaged 2 s")
ax.bar(x + w / 2, [S["controlled"][n]["overall"]["flagged_outside"] for n in names], width=w, color=CAT[1], label="outside it")
ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names]); ax.set_ylim(0, 1); ax.set_ylabel("share of frames flagged")
ax.legend(frameon=False); _label(ax, "(a) Does the detector find the damage? 60 LibriSpeech clips")
ax = axes[1]; w = 0.8 / len(ASR)
for i, m in enumerate(ASR):
    v = [S["focus"][n][m]["mean"] for n in names]; e = [[S["focus"][n][m]["mean"] - S["focus"][n][m]["ci"][0] for n in names], [S["focus"][n][m]["ci"][1] - S["focus"][n][m]["mean"] for n in names]]
    ax.bar(x + (i - 1) * w, v, width=w, color=CAT[i], label=m, yerr=e, capsize=3, error_kw=dict(ecolor=TXT2, lw=1))
ax.axhline(1, color=TXT2, lw=1, ls="--"); ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names])
ax.set_ylabel("evidence on flagged audio / share flagged"); ax.legend(frameon=False, title="speech-to-text model", title_fontsize=11)
_label(ax, "(b) Noise-focus ratio on the 100 VoxCeleb clips, with 95 % intervals")
for a_ in axes:
    for s in a_.spines.values():
        s.set_color(BORDER)
fig.suptitle("Same question, three detector backbones: whisper-small (AADD's), whisper-base, moonshine-base", x=0.07, ha="left", fontsize=13, fontweight="bold", color=TXT, y=0.96)
fig.savefig(FIG / "detector_backbones.png", dpi=DPI); fig.savefig(FIG / "detector_backbones.svg")
print("figure written")
