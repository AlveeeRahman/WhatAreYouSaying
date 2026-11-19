"""Representation erasure on audio (Li, Monroe and Jurafsky, 2016, applied to the input spectrogram).

Li et al. erase one input dimension by setting it to zero and measure the relative change in the
model's negative log-likelihood of the correct label. Here the "dimension" is a time x frequency
patch of the short-time Fourier transform (STFT) of the waveform. Erasing a patch means setting its
complex STFT coefficients to zero and resynthesising the waveform. The erased waveform is then given
to the speech model through its own front end, so the same operator serves a log-mel model
(Whisper) and a raw-waveform model (Moonshine), and it also serves a black-box server because the
erased audio can simply be sent over HTTP.

STFT parameters are Whisper's (window 400 samples = 25 ms, hop 160 samples = 10 ms, Hann window,
16 kHz), so a 100 ms patch is exactly 10 STFT frames and the frequency bands line up with Whisper's
80 slaney-mel filters (band b = mel filters 10b .. 10b+9).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SR = 16000
N_FFT = 400
HOP = 160
N_BINS = N_FFT // 2 + 1  # 201 bins, 40 Hz apart
BIN_HZ = SR / N_FFT

_WINDOWS: dict = {}


def _window(device):
    key = str(device)
    if key not in _WINDOWS:
        _WINDOWS[key] = torch.hann_window(N_FFT, device=device)
    return _WINDOWS[key]


def band_edges_hz(n_bands: int = 8, n_mels: int = 80, fmax: float = SR / 2) -> np.ndarray:
    """Lower edges of ``n_bands`` groups of Whisper's ``n_mels`` slaney-mel filters, plus fmax.

    The mel filterbank has ``n_mels + 2`` edge frequencies; filter i spans edges i .. i+2.
    Group b holds filters ``b*m .. (b+1)*m - 1`` (m = n_mels // n_bands), so its lower edge is edge
    ``b*m``. Groups are disjoint and together cover 0 .. fmax, so band shares sum to one.
    """
    import librosa

    assert n_mels % n_bands == 0, (n_mels, n_bands)
    m = n_mels // n_bands
    mel_f = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=0.0, fmax=fmax, htk=False)
    edges = [float(mel_f[b * m]) for b in range(n_bands)] + [float(fmax)]
    return np.asarray(edges, dtype=np.float64)


def band_bins(n_bands: int = 8) -> list[tuple[int, int]]:
    """[lo_bin, hi_bin) STFT bin ranges per band, disjoint and covering all 201 bins."""
    edges = band_edges_hz(n_bands)
    bins = [int(round(e / BIN_HZ)) for e in edges]
    bins[0], bins[-1] = 0, N_BINS
    return [(bins[b], bins[b + 1]) for b in range(n_bands)]


@dataclass(frozen=True)
class PatchGrid:
    """The erasure grid for one clip: ``n_t`` patches of ``patch_s`` seconds x ``n_bands`` bands."""

    n_samples: int
    patch_s: float = 0.1
    n_bands: int = 8

    @property
    def n_frames(self) -> int:  # torch.stft with center=True
        return self.n_samples // HOP + 1

    @property
    def frames_per_patch(self) -> int:
        f = self.patch_s * SR / HOP
        assert abs(f - round(f)) < 1e-9, "patch_s must be a multiple of 10 ms"
        return int(round(f))

    @property
    def n_t(self) -> int:
        return int(np.ceil(self.n_samples / SR / self.patch_s))

    @property
    def duration(self) -> float:
        return self.n_samples / SR

    @property
    def bins(self) -> list[tuple[int, int]]:
        return band_bins(self.n_bands)

    @property
    def edges_hz(self) -> np.ndarray:
        return band_edges_hz(self.n_bands)

    def patches(self) -> list[tuple[int, int]]:
        return [(b, t) for b in range(self.n_bands) for t in range(self.n_t)]

    def frame_range(self, t: int) -> tuple[int, int]:
        f = self.frames_per_patch
        return t * f, min((t + 1) * f, self.n_frames)

    def centre_s(self, t: int) -> float:
        return (t + 0.5) * self.patch_s


def stft(x: torch.Tensor) -> torch.Tensor:
    """x [..., N] float -> complex [..., 201, N // 160 + 1]."""
    return torch.stft(x.float(), N_FFT, HOP, window=_window(x.device), center=True, return_complex=True)


def istft(X: torch.Tensor, n_samples: int) -> torch.Tensor:
    return torch.istft(X, N_FFT, HOP, window=_window(X.device), center=True, length=n_samples)


def erase(wav: torch.Tensor, grid: PatchGrid, patches: list[tuple[int, int]], n_iter: int = 3) -> torch.Tensor:
    """Erase one patch per output row. wav [N] -> [len(patches), N].

    Patch (b, t) has its STFT coefficients set to zero in bins ``grid.bins[b]`` and frames
    ``grid.frame_range(t)``; everything else is left untouched and the waveform is resynthesised
    with the inverse STFT of the same window and hop.

    A single zero-and-resynthesise step is not exact: the 25 ms analysis windows overlap by 75 %,
    so the resynthesised signal's own STFT regains some energy in the patch from the neighbouring
    frames and bins (the usual STFT consistency problem). ``n_iter`` alternating projections
    (zero the patch, resynthesise, re-analyse) tighten it; see tests/test_erasure.py and the
    README for the measured suppression. The waveform outside the patch's time span is unchanged.
    """
    assert wav.ndim == 1 and len(wav) == grid.n_samples
    if not patches:
        return wav.new_zeros((0, grid.n_samples))
    X = stft(wav)
    out = X.unsqueeze(0).repeat(len(patches), 1, 1)
    y = None
    for it in range(max(1, n_iter)):
        if it > 0:
            out = stft(y)
        for i, (b, t) in enumerate(patches):
            lo, hi = grid.bins[b]
            f0, f1 = grid.frame_range(t)
            out[i, lo:hi, f0:f1] = 0
        y = istft(out, grid.n_samples)
    return y


def patch_energy(wav: torch.Tensor, grid: PatchGrid, b: int, t: int) -> float:
    """Sum of |STFT|^2 inside patch (b, t). Used by the tests to verify the operator."""
    X = stft(wav)
    lo, hi = grid.bins[b]
    f0, f1 = grid.frame_range(t)
    return float((X[lo:hi, f0:f1].abs() ** 2).sum())
