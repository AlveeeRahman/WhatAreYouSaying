"""The attribution formulas, checked against a synthetic backend whose surprisal is a known function
of patch energy, so the expected answer can be written down. CPU only."""
import numpy as np
import torch

from ways import erasure
from ways.attribution import explain
from ways.backends.base import Backend, Transcript
from ways.metrics import bootstrap_ci, paired_bootstrap, token_mass, token_summary, wer
from ways.quality import flag_profile


class EnergyBackend(Backend):
    """Token k "listens" to patch (b_k, t_k): S_k(x) = 1 + c * (E_ref - E_{b_k,t_k}(x)) / E_ref, so erasing
    its patch raises surprisal by exactly c, and erasing any other patch changes nothing. argmax flips
    when the patch is gone."""
    name = "synthetic"
    model_id = "energy"
    supports_surprisal = True
    device = "cpu"

    def __init__(self, grid, targets, c=2.0):
        self.grid, self.targets, self.c = grid, targets, c
        self.e_ref = None

    def transcribe(self, wav):
        K = len(self.targets)
        return Transcript(ids=np.arange(K + 2), keep=np.arange(K), pieces=[f" w{k}" for k in range(K)], text="")

    def surprisal(self, wavs, tr):
        B, K = len(wavs), len(self.targets)
        S = np.zeros((B, K)); arg = np.zeros((B, K), int)
        for i, w in enumerate(wavs):
            x = torch.as_tensor(np.asarray(w, np.float32))
            for k, (b, t) in enumerate(self.targets):
                e = erasure.patch_energy(x, self.grid, b, t)
                if self.e_ref is None:
                    self.e_ref = {}
                self.e_ref.setdefault(k, e)
                S[i, k] = 1 + self.c * (self.e_ref[k] - e) / self.e_ref[k]
                arg[i, k] = k + 1 if e > 0.5 * self.e_ref[k] else -1
        return S, arg


def _signal(n=16000):
    t = np.arange(n) / 16000
    rng = np.random.default_rng(1)
    return (0.3 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t) + 0.02 * rng.standard_normal(n)).astype(np.float32)


def test_explain_recovers_the_patch_each_token_used():
    x = _signal()
    grid = erasure.PatchGrid(len(x), 0.1, 8)
    targets = [(0, 2), (5, 7)]   # 300 Hz lies in band 0 (0-372 Hz), 3000 Hz in band 5 (2433-3571 Hz)
    att = explain(EnergyBackend(grid, targets), x, patch_s=0.1, n_bands=8, batch=16)
    assert att.mode == "surprisal" and att.delta.shape == (2, 8, grid.n_t)
    for k, (b, t) in enumerate(targets):
        assert np.unravel_index(att.delta[k].argmax(), att.delta[k].shape) == (b, t)
        assert abs(att.delta[k, b, t] - 2.0) < 0.1           # c = 2 nats, up to the residual the projection leaves
        assert att.flip[k, b, t] and att.flip[k].sum() == 1   # selection flips only there
        assert abs(att.rel[k, b, t] - 2.0 / 1.0) < 0.1        # rel = delta / S0 with S0 = 1
        other = np.delete(att.delta[k].ravel(), b * grid.n_t + t)
        # neighbours in time share a window edge with the erased patch, so they lose up to ~10 % of
        # its effect (measured 0.21 of c = 2); every other patch is numerically unchanged
        assert np.abs(other).max() < 0.15 * 2.0
    m = token_mass(att)
    assert np.allclose(m.sum((1, 2)), 1.0)
    rows = token_summary(att)
    assert [(r["top_band"], r["top_t"]) for r in rows] == targets
    assert all(abs(sum(r[f"band{b}"] for b in range(8)) - 1) < 1e-9 for r in rows)


def test_focus_ratio_is_one_when_mass_ignores_the_flags():
    T = 10
    probs = np.zeros((T * 5, 12), np.float32)
    probs[: T * 5 // 2, 0] = 0.9      # first half flagged (noise)
    f = flag_profile(probs, T, 0.1)
    assert f.tolist() == [1.0] * 5 + [0.0] * 5
    # uniform mass over time -> focus ratio 1; all mass in flagged half -> 2; in clean half -> 0
    uniform = np.full(T, 1 / T)
    assert abs((uniform * f).sum() / f.mean() - 1) < 1e-9
    flagged = np.r_[np.full(5, 0.2), np.zeros(5)]
    assert abs((flagged * f).sum() / f.mean() - 2) < 1e-9


def test_wer_counts():
    r = wer("the cat sat on the mat", "the cat sat on mat")
    assert (r["S"], r["D"], r["I"], r["N"]) == (0, 1, 0, 6) and abs(r["wer"] - 1 / 6) < 1e-9
    assert wer("hello world", "")["wer"] == 1.0
    assert wer("Hello, World!", "hello world")["wer"] == 0.0


def test_bootstrap_is_seeded_and_covers_the_mean():
    x = np.random.default_rng(0).normal(0.5, 0.1, 40)
    a, b = bootstrap_ci(x), bootstrap_ci(x)
    assert a == b and a["ci"][0] < 0.5 < a["ci"][1]
    p = paired_bootstrap(x + 0.2, x)
    assert p["ci"][0] > 0.19 and p["ci"][1] < 0.21 and p["n_pairs"] == 40
