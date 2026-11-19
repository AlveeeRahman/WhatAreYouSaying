"""Per-token erasure attribution.

Notation. x is the waveform, y = (y_1 .. y_K) the model's own greedy transcript, and
S_k(x) = -log p(y_k | y_<k, x) the surprisal of token k under teacher forcing. For patch P of the
STFT grid, x\\P is the waveform with P erased (ways.erasure.erase).

  delta_k(P) = S_k(x\\P) - S_k(x)              nats; > 0 means P supported token k
  rel_k(P)   = delta_k(P) / S_k(x)             Li et al. (2016) eq. 1, relative change of the
                                               negative log-likelihood, sign chosen so that a
                                               supporting patch is positive
  flip_k(P)  = [argmax_v p(v | y_<k, x\\P) != y_k]   the token selection changes

A black-box backend has no S_k, so it is scored at word level by re-decoding x\\P and aligning the
new transcript to y with jiwer: changed_w(P) = 1 if word w is not inside an "equal" alignment chunk.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import erasure
from .backends import Backend, Transcript


@dataclass
class Attribution:
    model: str
    backend: str
    mode: str                      # "surprisal" (white box) or "selection" (black box)
    duration: float
    patch_s: float
    n_bands: int
    band_edges_hz: list[float]
    pieces: list[str]              # scored units: tokens (surprisal) or words (selection)
    token_ids: list[int]
    times: list[float] | None      # per unit, seconds, if the backend gives timestamps
    surprisal0: np.ndarray         # [K] S_k(x); zeros in selection mode
    delta: np.ndarray              # [K, B, T] delta_k(P) (surprisal) or changed_w(P) in {0,1} (selection)
    rel: np.ndarray                # [K, B, T] rel_k(P); nan in selection mode
    flip: np.ndarray               # [K, B, T] bool
    text: str
    words: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    n_forward: int = 0

    @property
    def n_t(self) -> int:
        return self.delta.shape[2]

    def save(self, path):
        meta = {k: v for k, v in asdict(self).items() if k not in ("surprisal0", "delta", "rel", "flip")}
        np.savez_compressed(path, surprisal0=self.surprisal0.astype(np.float32), delta=self.delta.astype(np.float32),
                            rel=self.rel.astype(np.float32), flip=self.flip.astype(bool), meta=json.dumps(meta))

    @classmethod
    def load(cls, path) -> Attribution:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(surprisal0=z["surprisal0"], delta=z["delta"], rel=z["rel"], flip=z["flip"], **meta)


def explain(backend: Backend, wav: np.ndarray, patch_s: float = 0.1, n_bands: int = 8, batch: int = 64,
            transcript: Transcript | None = None, device: str | None = None, log=None) -> Attribution:
    """Run the erasure test over every patch of the grid for one clip."""
    t_start = time.time()
    wav = np.asarray(wav, np.float32)
    dev = device or getattr(backend, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    grid = erasure.PatchGrid(len(wav), patch_s, n_bands)
    tr = transcript or backend.transcribe(wav)
    x = torch.as_tensor(wav, device=dev)
    patches = grid.patches()
    K = tr.n_tokens
    if backend.supports_surprisal:
        mode = "surprisal"
        S0, _ = backend.surprisal(wav[None], tr)
        S0 = S0[0]
        delta = np.zeros((K, n_bands, grid.n_t), np.float32)
        flip = np.zeros((K, n_bands, grid.n_t), bool)
        yk = tr.token_ids()
        n_fwd = 1
        for i in range(0, len(patches), batch):
            chunk = patches[i:i + batch]
            xe = erasure.erase(x, grid, chunk).cpu().numpy()
            S, arg = backend.surprisal(xe, tr)
            n_fwd += len(chunk)
            for j, (b, t) in enumerate(chunk):
                delta[:, b, t] = S[j] - S0
                flip[:, b, t] = arg[j] != yk
            if log:
                log(f"  {backend.model_id}: {min(i + batch, len(patches))}/{len(patches)} patches")
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(S0[:, None, None] > 0, delta / S0[:, None, None], np.nan).astype(np.float32)
        pieces, ids, times = tr.pieces, [int(v) for v in yk], (tr.times.tolist() if tr.times is not None else None)
    else:
        import jiwer
        mode = "selection"
        words = [w["word"] for w in tr.group_words()]
        K = len(words)
        S0 = np.zeros(K, np.float32)
        delta = np.zeros((K, n_bands, grid.n_t), np.float32)
        ref = " ".join(_norm(w) for w in words)
        n_fwd = 1
        for i in range(0, len(patches), batch):
            chunk = patches[i:i + batch]
            xe = erasure.erase(x, grid, chunk).cpu().numpy()
            hyps = backend.transcribe_batch(xe)
            n_fwd += len(chunk)
            for j, (b, t) in enumerate(chunk):
                changed = np.ones(K, bool)
                hyp = " ".join(_norm(w["word"]) for w in hyps[j].group_words())
                if ref.strip() and hyp.strip():
                    o = jiwer.process_words(ref, hyp)
                    for ch in o.alignments[0]:
                        if ch.type == "equal":
                            changed[ch.ref_start_idx:ch.ref_end_idx] = False
                delta[:, b, t] = changed
            if log:
                log(f"  {backend.model_id}: {min(i + batch, len(patches))}/{len(patches)} patches")
        flip = delta.astype(bool)
        rel = np.full_like(delta, np.nan)
        pieces, ids, times = [" " + w for w in words], list(range(K)), None
    att = Attribution(model=backend.model_id, backend=backend.name, mode=mode, duration=grid.duration,
                      patch_s=patch_s, n_bands=n_bands, band_edges_hz=grid.edges_hz.tolist(), pieces=pieces,
                      token_ids=ids, times=times, surprisal0=np.asarray(S0, np.float32), delta=delta, rel=rel,
                      flip=flip, text=tr.text, words=tr.group_words(), seconds=time.time() - t_start, n_forward=n_fwd)
    return att


def _norm(s: str) -> str:
    import re
    s = s.lower().replace("—", " ").replace("–", " ")
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()
