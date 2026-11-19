"""The erasure operator, checked numerically. Runs on CPU in seconds."""
import numpy as np
import pytest
import torch

from ways import erasure
from ways.erasure import PatchGrid, erase, patch_energy, stft


def _speechlike(n=16000 * 2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    x += 0.2 * np.sin(2 * np.pi * 1800 * t) + 0.05 * rng.standard_normal(n)
    return torch.as_tensor(x, dtype=torch.float32)


def test_bands_are_disjoint_and_cover_all_bins():
    bins = erasure.band_bins(8)
    assert bins[0][0] == 0 and bins[-1][1] == erasure.N_BINS
    for (a, b), (c, _d) in zip(bins[:-1], bins[1:], strict=True):
        assert b == c and b > a
    edges = erasure.band_edges_hz(8)
    assert edges[0] == 0 and edges[-1] == 8000 and np.all(np.diff(edges) > 0)


def test_grid_geometry():
    g = PatchGrid(16000 * 3 + 50, 0.1, 8)
    assert g.frames_per_patch == 10
    assert g.n_t == 31 and g.n_frames == (16000 * 3 + 50) // 160 + 1
    assert g.frame_range(30)[1] == g.n_frames
    with pytest.raises(AssertionError):
        _ = PatchGrid(16000, 0.015).frames_per_patch


def test_istft_inverts_stft():
    x = _speechlike()
    y = erasure.istft(stft(x), len(x))
    assert torch.allclose(x, y, atol=1e-5)


def test_erase_empty_list_gives_empty_batch():
    x = _speechlike()
    assert erase(x, PatchGrid(len(x)), []).shape == (0, len(x))


def test_erase_suppresses_a_tone_in_the_patch_band():
    """A 550 Hz tone lies inside band 1 (372-745 Hz). Erasing patch (1, 7) must remove it there while
    the neighbouring band-0 patch keeps its (near zero) energy and the waveform outside the patch's
    time span is untouched. The suppression is limited by the 75 % window overlap: measured 19 dB
    after one projection and more after three (see README, "What erasing a patch does")."""
    n = 16000 * 2
    t = np.arange(n) / 16000
    x = torch.as_tensor(0.3 * np.sin(2 * np.pi * 550 * t), dtype=torch.float32)
    g = PatchGrid(len(x), 0.1, 8)
    b, t_ = 1, 7
    e0 = patch_energy(x, g, b, t_)
    y = erase(x, g, [(b, t_)])[0]
    e1 = patch_energy(y, g, b, t_)
    assert 10 * np.log10(e0 / e1) >= 15, f"only {10 * np.log10(e0 / e1):.1f} dB suppression"
    f0, f1 = g.frame_range(t_)
    s0, s1 = max(0, (f0 - 2) * erasure.HOP), min(len(x), (f1 + 2) * erasure.HOP)
    keep = torch.ones(len(x), dtype=torch.bool)
    keep[s0:s1] = False
    assert torch.allclose(x[keep], y[keep], atol=1e-5)


def test_erase_white_noise_every_band():
    rng = np.random.default_rng(0)
    x = torch.as_tensor(0.1 * rng.standard_normal(16000 * 2), dtype=torch.float32)
    g = PatchGrid(len(x), 0.1, 8)
    for b in range(8):
        e0, e1 = patch_energy(x, g, b, 7), patch_energy(erase(x, g, [(b, 7)])[0], g, b, 7)
        assert 10 * np.log10(e0 / e1) >= 12, (b, 10 * np.log10(e0 / e1))


def test_erased_patch_drops_below_the_signal_background_in_log_mel():
    """What the model sees: Whisper-style log-mel (80 slaney mels, floor = max - 80 dB). After erasure
    the patch must sit below the signal's own 10th-percentile level, i.e. quieter than its background."""
    import librosa
    rng = np.random.default_rng(3)
    n = 16000 * 2
    t = np.arange(n) / 16000
    f0 = 140 * (1 + 0.15 * np.sin(2 * np.pi * 2.5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / 16000
    x = sum(np.sin(h * phase) / h for h in range(1, 25)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) ** 2
    x = (0.2 * x + 0.003 * rng.standard_normal(n)).astype(np.float32)
    x = torch.as_tensor(x)
    g = PatchGrid(len(x), 0.1, 8)
    nf = len(x) // erasure.HOP
    fb = torch.tensor(librosa.filters.mel(sr=16000, n_fft=400, n_mels=80), dtype=torch.float32)
    lm = lambda X: torch.clamp(fb @ (X[..., :nf].abs() ** 2), min=1e-10).log10()
    L0 = lm(stft(x))
    floor = L0.max() - 8
    bg = 10 * (np.percentile(L0.numpy(), 10) - floor.item())
    sel = [(b, tt) for b in range(8) for tt in (4, 9, 14)]
    LE = lm(stft(erase(x, g, sel)))
    for j, (b, tt) in enumerate(sel):
        a, c = g.frame_range(tt)
        before = 10 * (L0[b * 10:(b + 1) * 10, a:c].mean() - floor).item()
        after = 10 * (LE[j][b * 10:(b + 1) * 10, a:c].mean() - floor).item()
        if before > bg + 6:
            assert after < bg, (b, tt, before, after, bg)
            assert before - after >= 15, (b, tt, before, after)


def test_erase_is_batched_consistently():
    x = _speechlike()
    g = PatchGrid(len(x), 0.1, 8)
    ps = [(0, 0), (3, 5), (7, 19)]
    batch = erase(x, g, ps)
    for i, p in enumerate(ps):
        single = erase(x, g, [p])[0]
        assert torch.allclose(batch[i], single)
