"""The universal figure runs on synthetic inputs with no model, no detector and no file system beyond a temp dir."""
import numpy as np

from ways.figure import degraded_spans, from_attribution, logmel, verdict, ways_figure


def test_logmel_shape_and_floor():
    wav = np.random.default_rng(0).standard_normal(16000 * 2).astype(np.float32) * 0.1
    L = logmel(wav)
    assert L.shape == (80, 200) and L.max() - L.min() <= 8.0 + 1e-6


def test_degraded_spans_merge_small_gaps():
    q = np.zeros((100, 2), np.float32)
    q[10:30, 0] = 0.9; q[33:50, 1] = 0.7; q[80:90, 0] = 0.6
    assert degraded_spans(q, 0.02) == [(0.2, 1.0), (1.6, 1.8)]
    assert degraded_spans(None, 0.02) == []


def test_verdict_counts_against_chance():
    assert verdict([True, False, True, None], 0.5) == "2 of 3 words on degraded audio, 2 expected by chance (50% of the clip is degraded)"
    assert verdict([False, False], 0.0).startswith("no degraded audio")
    assert verdict([True], 1.0).startswith("every frame degraded")


def test_figure_draws_from_plain_arrays(tmp_path):
    rng = np.random.default_rng(1)
    t = np.arange(16000 * 3) / 16000
    wav = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    quality = np.zeros((150, 3), np.float32); quality[40:90, 1] = 0.8          # 0.8 to 1.8 s "reverb"
    models = [dict(name="any-asr", tokens=["hello", "world", "!"], top=[(1, 3), (2, 12), None], patch_s=0.1, n_bands=8, wer=0.0),
              dict(name="other", tokens=["hi"], top=[(0, 20)], patch_s=0.1, n_bands=8)]
    out = tmp_path / "fig.png"
    fig = ways_figure(wav, 16000, quality=quality, quality_hop=0.02, quality_classes=["noise", "reverb", "hum"], models=models,
                      window=(0.8, 1.8), title="synthetic", out=out)
    assert out.exists() and out.with_suffix(".svg").exists() and fig is not None
    # a one-column detector and no models also work
    fig2 = ways_figure(wav, 16000, quality=quality.max(1), quality_hop=0.02, out=tmp_path / "fig2.png")
    assert (tmp_path / "fig2.png").exists()


def test_from_attribution_matches_argmax():
    class Att:
        delta = np.zeros((2, 8, 5), np.float32); pieces = [" a", " b"]; model = "m"; patch_s = 0.1; n_bands = 8
    Att.delta[0, 3, 2] = 1.0
    m = from_attribution(Att())
    assert m["top"] == [(3, 2), None] and m["name"] == "m" and m["tokens"] == [" a", " b"]
