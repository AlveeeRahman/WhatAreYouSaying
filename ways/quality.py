"""Audio quality context from AADD (Gertel, 2026; github.com/gertelrina/aadd, MIT).

AADD v3 gives, every 20 ms, the probability of each of 12 degradations. WAYS uses it unchanged: a
frame is "flagged" when any class probability is at or above 0.5 (AADD's default threshold), and
the flagged mask is what the noise-focus ratio in ways.metrics is computed against. The class
order below is AADD's.
"""
from __future__ import annotations

import numpy as np

SEG_S = 0.02
CLASSES = ["noise", "music", "reverb", "clipping", "lowpass", "highpass", "codec", "packet_loss",
           "hum", "clicks", "pitch_shift", "time_stretch"]


class Quality:
    """Thin wrapper over aadd.infer.Detector with the v3 frame model at 20 ms resolution."""

    def __init__(self, model: str = "v3", device: str | None = None, threshold: float = 0.5):
        from aadd.infer import Detector

        self.det = Detector(model=model, device=device, threshold=threshold, resolution=SEG_S)
        self.classes = list(self.det.classes)
        self.threshold = threshold
        self.model = model

    def frames(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        """-> [n_frames, n_classes] probabilities on the 20 ms grid (n_frames = ceil(len / 320))."""
        res = self.det.predict(np.asarray(wav, np.float32), sr)
        return np.asarray(res.frame_probs, np.float32)


def load_scan_csv(path, classes=CLASSES) -> dict[str, np.ndarray]:
    """Read an `aadd scan --segments --resolution 0.02` CSV into {clip: [n_frames, 12]}."""
    import pandas as pd

    df = pd.read_csv(path)
    df["clip"] = df["path"].map(lambda p: str(p).rsplit("/", 1)[-1].rsplit(".", 1)[0])
    cols = [f"p_{c}" for c in classes]
    return {k: g.sort_values("start_s")[cols].to_numpy(np.float32) for k, g in df.groupby("clip")}


def flagged_mask(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probs) >= threshold).any(1)


def intervals(probs: np.ndarray, threshold: float = 0.5, min_s: float = 0.1, classes=CLASSES):
    """AADD-style degraded intervals: (class, start_s, end_s, peak) for runs of frames >= threshold."""
    out = []
    for j, name in enumerate(classes):
        on = probs[:, j] >= threshold
        i = 0
        while i < len(on):
            if on[i]:
                k = i
                while k < len(on) and on[k]:
                    k += 1
                if (k - i) * SEG_S >= min_s:
                    out.append((name, i * SEG_S, k * SEG_S, float(probs[i:k, j].max())))
                i = k
            else:
                i += 1
    return sorted(out, key=lambda r: (r[1], r[0]))


def flag_profile(probs: np.ndarray | None, n_t: int, patch_s: float = 0.1, threshold: float = 0.5) -> np.ndarray | None:
    """Fraction of flagged 20 ms frames inside each attribution patch -> f [n_t] in [0, 1]."""
    if probs is None:
        return None
    flag = flagged_mask(probs, threshold).astype(np.float32)
    per = int(round(patch_s / SEG_S))
    f = np.zeros(n_t, np.float32)
    for t in range(n_t):
        seg = flag[t * per:(t + 1) * per]
        f[t] = seg.mean() if len(seg) else 0.0
    return f
