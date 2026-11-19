"""The same noise-focus question under three detectors whose backbones differ:
  whisper-small   AADD v3.1 as shipped (gertelrina/aadd, 88.6 M parameters)
  whisper-base    AADD's own frame model on the whisper-base encoder, weights from experiments/train_detectors.py
  moonshine-base  the same head on the Moonshine encoder, weights from experiments/train_detectors.py
Each detector scores (a) the 60 controlled clips, where the damaged window is known, so its hit rate inside and
outside the window is measured against ground truth, and (b) the 100 VoxCeleb clips, where its flags feed the
noise-focus ratio of each speech-to-text model. The attribution maps never change; only the flags do.
Writes experiments/results/detector_backbones.{json,md}; the figure is built from the json by
docs/images/src/make_detector_figure.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AUDIO_DIR, data_root  # noqa: E402
from train_detectors import OUT as DET_OUT  # noqa: E402
from train_detectors import frame_probs, load_detector  # noqa: E402

from ways.metrics import bootstrap_ci, paired_bootstrap  # noqa: E402
from ways.provenance import sha256  # noqa: E402
from ways.quality import CLASSES as AADD_CLASSES  # noqa: E402

AUDIO = data_root(sys.argv[1] if len(sys.argv) > 1 else None) / AUDIO_DIR
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
    yield "whisper-small detector (AADD v3.1)", lambda w: np.asarray(v3.predict(w, 16000).frame_probs, np.float32), dict(n_params=int(sum(p.numel() for p in v3.model.parameters())), weights="AADD v3.1 release")
    for name in ("whisper-base", "moonshine-base"):
        m, ck = load_detector(name, DEV)
        yield f"{name} detector", (lambda w, m=m: frame_probs(m, w, DEV)), dict(n_params=int(sum(p.numel() for p in m.parameters())), weights="AADD recipe on LibriSpeech dev-clean", val_auc=ck["meta"]["val_auc"], steps=ck["meta"]["step"])


S = dict(detectors={}, controlled={}, focus={})
(RES / "aadd").mkdir(exist_ok=True)
for name, fn, info in detectors():
    key = name.split(" ")[0]
    S["detectors"][name] = info
    ddir = RES / "aadd" / f"det_{key}"; ddir.mkdir(exist_ok=True)
    ckpt = DET_OUT / f"{key}.pt"
    tag = sha256(ckpt) if ckpt.exists() else "aadd-v3.1-release"
    tag_f = ddir / "weights.sha256"
    if not tag_f.exists() or tag_f.read_text().strip() != tag:
        for old in ddir.glob("*.npy"):
            old.unlink()
        tag_f.write_text(tag + "\n")
    info["weights_sha256"] = tag
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
    for clip in sample.file:
        f = RES / "aadd" / f"det_{key}" / f"{clip}.npy"
        if not f.exists():
            w, _ = sf.read(AUDIO / f"{clip}.wav", dtype="float32"); np.save(f, fn(w))
    print(name, S["controlled"][name]["overall"], flush=True)

import compare  # noqa: E402  (loads the clips and tokens tables)

compare.DETS.update({f"det_{n.split(' ')[0]}": 0.02 for n in S["detectors"]})
for name in S["detectors"]:
    det = f"det_{name.split(' ')[0]}"
    S["focus"][name] = {m: bootstrap_ci(compare.focus_under(det, m)) for m in ASR}
    fb, fm = compare.focus_under(det, "whisper-base"), compare.focus_under(det, "moonshine-base")
    S["focus"][name]["paired whisper-base - moonshine-base"] = paired_bootstrap(fb, fm)
(RES / "detector_backbones.json").write_text(json.dumps(S, indent=1, default=float))

ci = lambda d, f="{:.2f}": f"{f.format(d['mean'])} [{f.format(d['ci'][0])}, {f.format(d['ci'][1])}]"
L = ["| detector | parameters | weights | flags inside the damaged 2 s | flags outside | right class inside |", "|---|---|---|---|---|---|"]
for name, info in S["detectors"].items():
    o = S["controlled"][name]["overall"]
    L.append(f"| {name} | {info['n_params'] / 1e6:.1f} M | {info['weights']} | {o['flagged_inside']:.2f} | {o['flagged_outside']:.2f} | {o['class_inside']:.2f} |")
L += ["", "| detector | " + " | ".join(ASR) + " | paired whisper-base − moonshine-base |", "|---|" + "---|" * (len(ASR) + 1)]
for name, d in S["focus"].items():
    L.append(f"| {name} | " + " | ".join(ci(d[m]) + f" (n={d[m]['n']})" for m in ASR) + f" | {ci(d['paired whisper-base - moonshine-base'], '{:+.3f}')} |")
(RES / "detector_backbones.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
