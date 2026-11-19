"""Noise in the latent space: what degraded or loud audio does to each encoder layer.

Two measures per layer, both on the 20 ms frame grid, for white-box backends (Whisper, Moonshine):

  spill-over   Erase the degraded stretches (any AADD class at or above 0.5, or a given window) with
               ways.erasure, run the encoder on the original and on the erased audio, and measure
               how much each layer's representation moves per frame:
                   shift_l(t) = || h_l(x)[t] - h_l(x \\ D)[t] || / || h_l(x)[t] ||
               Inside D it has to move. Outside D, in clean frames, it should not move if the
               layer kept clean speech separate from the noise; the mean there is the spill-over.
  grad-cam     Grad-CAM (Selvaraju et al.) on every encoder layer, time-only because encoder frames
               have no frequency axis: alpha_l = mean_t dS/dh_l[t], cam_l(t) = ReLU(sum_c alpha_l[c]
               h_l[t, c]), with S the summed surprisal of the model's own transcript under teacher
               forcing. Normalised per layer to mass 1. The AADD recipe of weighting all layers is why
               every layer is shown rather than the last one. This is the heuristic half: a gradient
               saliency, not an erasure test, and it is reported next to the spill-over for that reason.

Per layer the summary gives: cam_focus (CAM mass on degraded frames / degraded share, 1 = no
preference), rho_loud (Spearman correlation of the CAM with frame loudness in dB), shift_in and
shift_out (mean relative shift inside and outside D).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import erasure
from .backends import Backend, Transcript

SEG_S = 0.02


@dataclass
class LatentResult:
    model: str
    n_frames: int
    layers: int
    cam: np.ndarray            # [L, n_frames], each layer sums to 1 (zeros if the layer has no positive CAM)
    shift: np.ndarray          # [L, n_frames] relative latent shift under erasure of D
    loudness_db: np.ndarray    # [n_frames]
    degraded: np.ndarray       # [n_frames] bool, the erased frames D
    summary: list[dict]        # per layer: cam_focus, rho_loud, shift_in, shift_out
    text: str

    def to_dict(self):
        return dict(model=self.model, n_frames=self.n_frames, layers=self.layers, degraded_share=float(self.degraded.mean()),
                    summary=self.summary, text=self.text)


def loudness_db(wav: np.ndarray, n_frames: int) -> np.ndarray:
    hop = int(SEG_S * 16000)
    x = np.asarray(wav, np.float32)[: n_frames * hop]
    x = np.pad(x, (0, n_frames * hop - len(x)))
    return 20 * np.log10(np.sqrt((x.reshape(n_frames, hop) ** 2).mean(1)) + 1e-6)


def degraded_frames(probs, n_frames: int, threshold: float = 0.5, window=None) -> np.ndarray:
    """Frames to erase: AADD flags (any class >= threshold) or an explicit (start_s, end_s) window."""
    d = np.zeros(n_frames, bool)
    if window is not None:
        a, b = int(window[0] / SEG_S), int(np.ceil(window[1] / SEG_S))
        d[max(a, 0):min(b, n_frames)] = True
    elif probs is not None:
        q = np.asarray(probs); q = q[:, None] if q.ndim == 1 else q
        n = min(n_frames, len(q))
        d[:n] = (q[:n] >= threshold).any(1)
    return d


def erase_frames(wav: np.ndarray, frames: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """Erase every 100 ms patch, in all bands, whose frames are mostly degraded (the same operator as the attribution)."""
    grid = erasure.PatchGrid(len(wav), 0.1, 8)
    per = int(round(0.1 / SEG_S))
    patches = [(b, t) for t in range(grid.n_t) if frames[t * per:(t + 1) * per].mean() >= 0.5 for b in range(8)]
    if not patches:
        return np.asarray(wav, np.float32).copy()
    x = torch.as_tensor(np.asarray(wav, np.float32))
    X = erasure.stft(x); y = None
    for it in range(n_iter):
        if it:
            X = erasure.stft(y)
        for b, t in patches:
            lo, hi = grid.bins[b]; f0, f1 = grid.frame_range(t); X[lo:hi, f0:f1] = 0
        y = erasure.istft(X, len(x))
    return y.numpy()


def _resample(a: np.ndarray, n_out: int, dur: float) -> np.ndarray:
    """Encoder frames of any rate -> the 20 ms grid, linear interpolation along time; a is [T] or [T, d]."""
    src = np.arange(len(a)) * (dur / len(a)); dst = np.arange(n_out) * SEG_S
    if a.ndim == 1:
        return np.interp(dst, src, a)
    return np.stack([np.interp(dst, src, a[:, c]) for c in range(a.shape[1])], 1)


def encoder_cam_and_states(backend: Backend, wav: np.ndarray, tr: Transcript):
    """-> (cam [L, n20] un-normalised, states [L, n20, d]) for one waveform, from the backend's own encoder."""
    if not getattr(backend, "supports_surprisal", False):
        raise NotImplementedError(f"{backend.name} is a black-box backend: no encoder states over HTTP, so no latent view")
    dur = len(wav) / 16000; n20 = int(len(wav) // (SEG_S * 16000))
    seq = torch.as_tensor(tr.ids, device=backend.device); keep = torch.as_tensor(tr.keep, device=backend.device)
    if backend.name == "hf-whisper":
        feats, _ = backend.features(np.asarray(wav, np.float32)[None]); feats = feats.detach().requires_grad_(True)
        out = backend.model(input_features=feats, decoder_input_ids=seq[:-1][None], output_hidden_states=True)
    else:
        iv, am = backend.inputs(np.asarray(wav, np.float32)[None]); iv = iv.detach().requires_grad_(True)
        out = backend.model(input_values=iv, attention_mask=am, decoder_input_ids=seq[:-1][None], output_hidden_states=True)
    hs = out.encoder_hidden_states
    for h in hs:
        h.retain_grad()
    lp = torch.log_softmax(out.logits.float(), -1); tgt = seq[1:]; pos = torch.arange(len(tgt), device=backend.device)
    (-lp[0, pos, tgt][keep]).sum().backward()
    cams, states = [], []
    for h in hs:
        A, G = h[0].detach(), h.grad[0]
        cam = torch.relu((A * G.mean(0, keepdim=True)).sum(-1)).float().cpu().numpy()
        A = A.float().cpu().numpy()
        if backend.name == "hf-whisper":
            cam, A = cam[:n20], A[:n20]          # Whisper encoder frames are already 20 ms, padded to 30 s
        else:
            cam, A = _resample(cam, n20, dur), _resample(A, n20, dur)
        cams.append(cam); states.append(A)
    backend.model.zero_grad(set_to_none=True)
    return np.stack(cams), np.stack(states)


def layer_summary(cam: np.ndarray, shift: np.ndarray, degraded: np.ndarray, loud: np.ndarray) -> list[dict]:
    from scipy.stats import spearmanr

    out = []
    share = float(degraded.mean())
    for l in range(cam.shape[0]):
        c = cam[l]
        row = dict(layer=l, cam_mass=float(c.sum()))
        if c.sum() > 0 and 0 < share < 1:
            cn = c / c.sum(); row["cam_focus"] = float(cn[degraded].sum() / share)
        else:
            row["cam_focus"] = float("nan")
        row["rho_loud"] = float(spearmanr(c, loud).statistic) if c.std() > 0 and loud.std() > 0 else float("nan")
        row["shift_in"] = float(shift[l][degraded].mean()) if degraded.any() else float("nan")
        row["shift_out"] = float(shift[l][~degraded].mean()) if (~degraded).any() else float("nan")
        out.append(row)
    return out


def latent_view(backend: Backend, wav: np.ndarray, probs=None, window=None, threshold: float = 0.5,
                transcript: Transcript | None = None) -> LatentResult:
    wav = np.asarray(wav, np.float32)
    n20 = int(len(wav) // (SEG_S * 16000))
    tr = transcript or backend.transcribe(wav)
    degraded = degraded_frames(probs, n20, threshold, window)
    cam, states = encoder_cam_and_states(backend, wav, tr)
    if degraded.any():
        _, states_e = encoder_cam_and_states(backend, erase_frames(wav, degraded), tr)
        shift = np.linalg.norm(states - states_e, axis=2) / (np.linalg.norm(states, axis=2) + 1e-9)
    else:
        shift = np.zeros_like(cam)
    loud = loudness_db(wav, n20)
    cam_n = np.where(cam.sum(1, keepdims=True) > 0, cam / np.maximum(cam.sum(1, keepdims=True), 1e-12), 0.0)
    return LatentResult(model=backend.model_id, n_frames=n20, layers=cam.shape[0], cam=cam_n.astype(np.float32),
                        shift=shift.astype(np.float32), loudness_db=loud.astype(np.float32), degraded=degraded,
                        summary=layer_summary(cam, shift, degraded, loud), text=tr.text)
