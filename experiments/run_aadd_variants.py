"""Degradation probabilities for the 100 clips from three AADD detectors with different backbones:
  v3      Whisper-small encoder, 12 classes, 20 ms frames  (the main detector)
  v3-lite 1.3 M-parameter CRNN distilled from v3, 12 classes, 20 ms frames
  v2      EfficientNet-B0 on log-mel, 7 synthetic classes + None, 1 s segments
Each is run through the AADD Python API on the same waveforms and saved as
experiments/results/aadd/<name>/<clip>.npy ([n_frames, n_classes] on that detector's own grid).
The v3 scan CSV in the data folder is checked against the fresh v3 run on the first clip.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from common import AADD_CSV, AUDIO_DIR, data_root

from ways.quality import load_scan_csv

HERE = Path(__file__).resolve().parent
DATA = data_root(sys.argv[1] if len(sys.argv) > 1 else None, need=(AUDIO_DIR, AADD_CSV))
AUDIO = DATA / AUDIO_DIR
AADD_CSV = DATA / AADD_CSV
sample = pd.read_csv(HERE / "sample100.csv")
OUT = HERE / "results/aadd"
from aadd.infer import Detector

info = {}
for name in ["v3", "v3-lite", "v2"]:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    det = Detector(model=name, resolution=0.02 if name.startswith("v3") else 1.0)
    info[name] = dict(classes=det.classes, resolution_s=0.02 if name.startswith("v3") else 1.0,
                      checkpoint=det.model_name, n_params=int(sum(p.numel() for p in det.model.parameters())))
    t0 = time.time()
    for clip in sample.file:
        f = d / f"{clip}.npy"
        if f.exists():
            continue
        wav, sr = sf.read(AUDIO / f"{clip}.wav", dtype="float32")
        res = det.predict(wav, sr)
        probs = res.frame_probs if getattr(res, "frame_probs", None) is not None else res.probs
        np.save(f, np.asarray(probs, np.float32))
    info[name]["seconds"] = round(time.time() - t0, 1)
    print(name, info[name], flush=True)
# consistency with the scan CSV in the data folder (same v3 weights, CLI path)
csv = load_scan_csv(AADD_CSV)
c0 = sample.file[0]
a, b = np.load(OUT / "v3" / f"{c0}.npy"), csv[c0]
n = min(len(a), len(b))
info["v3_vs_scan_csv"] = dict(clip=c0, n_frames=[int(len(a)), int(len(b))],
                                 max_abs_diff=float(np.abs(a[:n] - b[:n]).max()),
                                 flag_agreement=float(((a[:n] >= 0.5).any(1) == (b[:n] >= 0.5).any(1)).mean()))
(OUT / "info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info["v3_vs_scan_csv"]))
