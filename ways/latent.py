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
    """The latent view of one clip. With L layers: cam is [L, n_frames] and each layer sums to 1
    (zeros if the layer has no positive CAM), shift is [L, n_frames] the relative latent shift under
    erasure of the degraded frames, loudness_db is [n_frames], degraded is [n_frames] bool, and
    summary has one dict per layer (cam_focus, rho_loud, shift_in, shift_out)."""
    model: str
    n_frames: int
    layers: int
    cam: np.ndarray
    shift: np.ndarray
    loudness_db: np.ndarray
    degraded: np.ndarray
    summary: list[dict]
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


def _forward(backend: Backend, wav: np.ndarray, tr: Transcript):
    """One teacher-forced pass with gradients enabled on the encoder input. -> (hidden states per layer,
    log-probs [1, positions, vocab], target ids, 20 ms frame count, duration in seconds)."""
    if not getattr(backend, "supports_surprisal", False):
        raise NotImplementedError(f"{backend.name} is a black-box backend: no encoder states over HTTP, so no latent view")
    dur = len(wav) / 16000; n20 = int(len(wav) // (SEG_S * 16000))
    seq = torch.as_tensor(tr.ids, device=backend.device)
    if backend.name == "hf-whisper":
        feats, _ = backend.features(np.asarray(wav, np.float32)[None]); feats = feats.detach().requires_grad_(True)
        out = backend.model(input_features=feats, decoder_input_ids=seq[:-1][None], output_hidden_states=True)
    else:
        iv, am = backend.inputs(np.asarray(wav, np.float32)[None]); iv = iv.detach().requires_grad_(True)
        out = backend.model(input_values=iv, attention_mask=am, decoder_input_ids=seq[:-1][None], output_hidden_states=True)
    hs = out.encoder_hidden_states
    for h in hs:
        h.retain_grad()
    lp = torch.log_softmax(out.logits.float(), -1)
    return hs, lp, seq[1:], n20, dur


def _cam_of(h: torch.Tensor, backend: Backend, n20: int, dur: float, pooled: bool = True) -> np.ndarray:
    """Grad-CAM of one layer from its activations and the gradient already on it, on the 20 ms grid.
    pooled=True is Grad-CAM proper (channel weights = the gradient averaged over time); pooled=False keeps
    the gradient per frame (gradient times activation, summed over channels), which keeps the where."""
    A, G = h[0].detach(), h.grad[0]
    W = G.mean(0, keepdim=True) if pooled else G
    cam = torch.relu((A * W).sum(-1)).float().cpu().numpy()
    return cam[:n20] if backend.name == "hf-whisper" else _resample(cam, n20, dur)


def encoder_cam_and_states(backend: Backend, wav: np.ndarray, tr: Transcript):
    """-> (cam [L, n20] un-normalised, states [L, n20, d]) for one waveform, from the backend's own encoder."""
    hs, lp, tgt, n20, dur = _forward(backend, wav, tr)
    keep = torch.as_tensor(tr.keep, device=backend.device); pos = torch.arange(len(tgt), device=backend.device)
    (-lp[0, pos, tgt][keep]).sum().backward()
    cams, states = [], []
    for h in hs:
        cams.append(_cam_of(h, backend, n20, dur))
        A = h[0].detach().float().cpu().numpy()
        states.append(A[:n20] if backend.name == "hf-whisper" else _resample(A, n20, dur))
    backend.model.zero_grad(set_to_none=True)
    return np.stack(cams), np.stack(states)


def token_cam(backend: Backend, wav: np.ndarray, tr: Transcript | None = None, pooled: bool = False):
    """A saliency map of every scored token on every encoder layer: one backward pass per token from that
    token's own surprisal, the same recipe as the clip-level CAM but not summed over the transcript.

    pooled=False (default) keeps the gradient per frame: gradient times activation summed over channels,
    Grad-CAM without the time pooling. pooled=True is Grad-CAM proper, with channel weights averaged over
    time. The pooled map carries no per-token time information on these encoders (every token's map sits
    in the middle of the clip, 1 to 5 s wide; see experiments/ambiguous_case.py), so the token view defaults
    to the per-frame map. Both are gradient saliencies and read as hints next to the erasure test.

    -> (tcam [K tokens, L layers, n20 frames], each (token, layer) row normalised to mass 1, zeros if the
    layer has no positive CAM for that token; the transcript used)
    """
    wav = np.asarray(wav, np.float32)
    tr = tr or backend.transcribe(wav)
    hs, lp, tgt, n20, dur = _forward(backend, wav, tr)
    keep = [int(k) for k in tr.keep]
    out = np.zeros((len(keep), len(hs), n20), np.float32)
    for i, k in enumerate(keep):
        for h in hs:
            h.grad = None
        (-lp[0, k, tgt[k]]).backward(retain_graph=(i < len(keep) - 1))
        for l, h in enumerate(hs):
            out[i, l] = _cam_of(h, backend, n20, dur, pooled)
    backend.model.zero_grad(set_to_none=True)
    mass = out.sum(2, keepdims=True)
    return np.where(mass > 0, out / np.maximum(mass, 1e-12), 0.0).astype(np.float32), tr


def token_cam_summary(tcam: np.ndarray, degraded: np.ndarray, pieces, hop: float = SEG_S,
                      erase_profile: np.ndarray | None = None, erase_hop: float = 0.1) -> list[dict]:
    """One row per token, read at the top layer unless noted. "CAM" here is whichever token map was given
    (the per-frame gradient map by default).

    cam_t         CAM-weighted mean time (s)
    cam_width_s   effective width, exp(entropy) frames times hop: 0.1 s is one sharp syllable, 2 s is a smear
    cam_in        share of the token's CAM inside the degraded frames (top layer); cam_in_layers lists every layer
    erase_in      share of the token's erasure mass inside the degraded stretch, when erase_profile [K, patches] is given
    erase_t       time (s) of the token's most important erasure patch
    reading       both methods put the token's evidence on the degraded audio ("degraded"), on the clean audio
                  ("clean"), or they disagree ("split"); "diffuse" when the CAM is wider than one second
    """
    K, L, n = tcam.shape
    t = (np.arange(n) + 0.5) * hop
    d = np.asarray(degraded, bool)[:n]
    rows = []
    for k in range(K):
        c = tcam[k, -1]
        row = dict(k=k, piece=str(pieces[k]))
        if c.sum() <= 0:
            row.update(cam_t=float("nan"), cam_width_s=float("nan"), cam_in=float("nan"), cam_in_layers=[float("nan")] * L)
        else:
            ent = -(c[c > 0] * np.log(c[c > 0])).sum()
            row.update(cam_t=float((c * t).sum()), cam_width_s=float(np.exp(ent) * hop), cam_in=float(c[d].sum()),
                       cam_in_layers=[float(tcam[k, l][d].sum()) if tcam[k, l].sum() > 0 else float("nan") for l in range(L)])
        if erase_profile is not None:
            e = np.asarray(erase_profile[k], np.float64)
            if np.isfinite(e).all() and e.sum() > 0:
                fe = np.array([d[int(j * erase_hop / hop): int((j + 1) * erase_hop / hop)].mean() if int((j + 1) * erase_hop / hop) > int(j * erase_hop / hop) else 0.0
                               for j in range(len(e))])
                row.update(erase_in=float((e / e.sum() * fe).sum()), erase_t=float((int(e.argmax()) + 0.5) * erase_hop))
            else:
                row.update(erase_in=float("nan"), erase_t=float("nan"))
        ci, ei = row.get("cam_in", float("nan")), row.get("erase_in", float("nan"))
        if np.isnan(ci):
            row["reading"] = "no CAM"
        elif row["cam_width_s"] > 1.0:
            row["reading"] = "diffuse"
        elif np.isnan(ei):
            row["reading"] = "degraded" if ci >= 0.5 else "clean"
        else:
            row["reading"] = "degraded" if (ci >= 0.5 and ei >= 0.5) else "clean" if (ci < 0.5 and ei < 0.5) else "split"
        rows.append(row)
    return rows


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
    from .attribution import _check_length

    wav = np.asarray(wav, np.float32)
    _check_length(backend, wav)
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
