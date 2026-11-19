"""The attribution formulas, checked against a synthetic backend whose surprisal is a known function
of patch energy, so the expected answer can be written down. CPU only."""
import numpy as np
import torch

from ways import erasure
from ways.attribution import Attribution, explain
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

    def __init__(self, grid, targets, c=2.0, base=1.0):
        self.grid, self.targets, self.c, self.base = grid, targets, c, base
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
                S[i, k] = self.base + self.c * (self.e_ref[k] - e) / self.e_ref[k]
                arg[i, k] = k + 1 if e > 0.5 * self.e_ref[k] else -1
        return S, arg


def _signal(n=16000):
    t = np.arange(n) / 16000
    rng = np.random.default_rng(1)
    return (0.3 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t) + 0.02 * rng.standard_normal(n)).astype(np.float32)


def test_explain_recovers_the_patch_each_token_used():
    x = _signal()
    grid = erasure.PatchGrid(len(x), 0.1, 8)
    targets = [(0, 2), (5, 7)]
    att = explain(EnergyBackend(grid, targets), x, patch_s=0.1, n_bands=8, batch=16)
    assert att.mode == "surprisal" and att.delta.shape == (2, 8, grid.n_t)
    for k, (b, t) in enumerate(targets):
        assert np.unravel_index(att.delta[k].argmax(), att.delta[k].shape) == (b, t)
        assert abs(att.delta[k, b, t] - 2.0) < 0.1
        assert att.flip[k, b, t] and att.flip[k].sum() == 1
        assert abs(att.rel[k, b, t] - 2.0 / 1.0) < 0.1
        other = np.delete(att.delta[k].ravel(), b * grid.n_t + t)
        assert np.abs(other).max() < 0.15 * 2.0
    m = token_mass(att)
    assert np.allclose(m.sum((1, 2)), 1.0)
    rows = token_summary(att)
    assert [(r["top_band"], r["top_t"]) for r in rows] == targets
    assert all(abs(sum(r[f"band{b}"] for b in range(8)) - 1) < 1e-9 for r in rows)


def test_focus_ratio_is_one_when_mass_ignores_the_flags():
    T = 10
    probs = np.zeros((T * 5, 12), np.float32)
    probs[: T * 5 // 2, 0] = 0.9
    f = flag_profile(probs, T, 0.1)
    assert f.tolist() == [1.0] * 5 + [0.0] * 5
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


def test_rel_divides_by_the_unerased_surprisal():
    wav = _signal()
    grid = erasure.PatchGrid(len(wav), 0.1, 8)
    att = explain(EnergyBackend(grid, [(0, 2), (5, 7)], c=2.0, base=2.5), wav, batch=16)
    for k, (b, t) in enumerate([(0, 2), (5, 7)]):
        assert abs(att.surprisal0[k] - 2.5) < 1e-6
        assert abs(att.rel[k, b, t] - att.delta[k, b, t] / 2.5) < 1e-6 and abs(att.rel[k, b, t] - 0.8) < 0.05


def _attribution(delta, patch_s=0.1):
    K, B, T = delta.shape
    return Attribution(model="m", backend="synthetic", mode="surprisal", duration=T * patch_s, patch_s=patch_s, n_bands=B,
                       band_edges_hz=[0.0] * (B + 1), pieces=[f" w{k}" for k in range(K)], token_ids=list(range(K)), times=None,
                       surprisal0=np.ones(K, np.float32), delta=delta.astype(np.float32), rel=delta.astype(np.float32),
                       flip=np.zeros(delta.shape, bool), text="")


def test_token_summary_focus_ratio_against_flags():
    T = 10
    probs = np.zeros((T * 5, 12), np.float32)
    probs[: T * 5 // 2, 0] = 0.9
    delta = np.zeros((3, 8, T))
    delta[0] = 1.0
    delta[1, :, :5] = 1.0
    delta[2, :, 5:] = 1.0
    rows = token_summary(_attribution(delta), probs)
    assert [round(r["focus_ratio"], 9) for r in rows] == [1.0, 2.0, 0.0]
    assert [round(r["mass_flagged"], 9) for r in rows] == [0.5, 1.0, 0.0]


def test_flag_profile_follows_time_for_any_patch_length():
    probs = np.zeros((500, 12), np.float32)
    probs[250:, 0] = 0.9
    for patch_s in (0.05, 0.1, 0.2, 0.3):
        f = flag_profile(probs, int(np.ceil(10 / patch_s)), patch_s)
        first = int(np.argmax(f > 0))
        assert abs(first * patch_s - 5.0) < patch_s + 1e-9, (patch_s, first)
        assert f[: first].sum() == 0 and f[first + 1: int(10 / patch_s)].min() == 1.0


def test_grid_size_and_band_limits():
    assert erasure.PatchGrid(33600, 0.3, 8).n_t == 7 and erasure.PatchGrid(33601, 0.3, 8).n_t == 8
    assert erasure.PatchGrid(16000, 0.1, 8).n_t == 10
    for n in (1, 2, 4, 8, 16, 40):
        bins = erasure.band_bins(n)
        assert bins[0][0] == 0 and bins[-1][1] == erasure.N_BINS and all(lo < hi for lo, hi in bins)
    try:
        erasure.band_bins(80)
    except ValueError:
        pass
    else:
        raise AssertionError("80 bands should be refused")


class TextBackend(Backend):
    """A black-box backend that answers `base` for the original clip and `other` for any erased one."""
    name = "text-only"
    model_id = "text"
    supports_surprisal = False
    device = "cpu"

    def __init__(self, base, other):
        self.base, self.other = base, other

    def _tr(self, text):
        words = text.split()
        return Transcript(ids=np.arange(len(words) + 2), keep=np.arange(len(words)), pieces=[" " + w for w in words], text=text)

    def transcribe(self, wav):
        return self._tr(self.base)

    def transcribe_batch(self, wavs):
        return [self._tr(self.other) for _ in wavs]


def test_word_change_map_skips_punctuation_only_words():
    wav = _signal(8000)
    att = explain(TextBackend("well — yes sir", "well — no sir"), wav, patch_s=0.1, n_bands=1)
    assert att.delta[:, 0, 0].tolist() == [0.0, 0.0, 1.0, 0.0]
    same = explain(TextBackend("... ...", "... ..."), wav, patch_s=0.1, n_bands=1)
    assert same.delta.sum() == 0


def test_a_clip_longer_than_the_model_window_is_refused():
    be = TextBackend("a", "a")
    be.max_seconds = 0.25
    try:
        explain(be, _signal(8000), n_bands=1)
    except ValueError as e:
        assert "at most" in str(e)
    else:
        raise AssertionError("a 0.5 s clip should be refused by a 0.25 s model")


def test_wer_substitution_and_insertion():
    r = wer("the cat sat", "the dog sat down")
    assert (r["S"], r["D"], r["I"], r["N"]) == (1, 0, 1, 3) and abs(r["wer"] - 2 / 3) < 1e-9
