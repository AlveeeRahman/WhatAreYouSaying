"""The universal figure as software: it draws from plain arrays, every text is editable, no two texts
overlap, and the same inputs give the same bytes."""
import hashlib

import numpy as np
import pytest

from ways.figure import (FigureStyle, FigureText, LatentText, LayoutError, degraded_spans, from_attribution,
                         latent_figure, logmel, text_overlaps, verdict, ways_figure, wrap_to_width)


def _wav(seconds=3, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(16000 * seconds) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _models():
    return [dict(name="any-asr", tokens=["hello", "world", "!"], top=[(1, 3), (2, 12), None], patch_s=0.1, n_bands=8, wer=0.0),
            dict(name="other", tokens=["hi"], top=[(0, 20)], patch_s=0.1, n_bands=8)]


def _quality():
    q = np.zeros((150, 3), np.float32); q[40:90, 1] = 0.8
    return q


def test_logmel_shape_and_floor():
    L = logmel(_wav(2))
    assert L.shape == (80, 200) and L.max() - L.min() <= 8.0 + 1e-6


def test_degraded_spans_merge_small_gaps():
    q = np.zeros((100, 2), np.float32)
    q[10:30, 0] = 0.9; q[33:50, 1] = 0.7; q[80:90, 0] = 0.6
    assert degraded_spans(q, 0.02) == [(0.2, 1.0), (1.6, 1.8)]
    assert degraded_spans(None, 0.02) == []


def test_verdict_counts_against_chance_and_is_editable():
    assert verdict([True, False, True, None], 0.5) == "2 of 3 words on degraded audio, 2 expected by chance (50% of the clip is degraded)"
    assert verdict([False, False], 0.0).startswith("no degraded audio")
    assert verdict([True], 1.0).startswith("every frame degraded")
    assert verdict([True, False], 0.5, FigureText(verdict_mixed="{k}/{K}")) == "1/2"


def test_wrap_to_width_respects_measured_width():
    s = "a fairly long title that should wrap into more than one line at a narrow width"
    w = wrap_to_width(s, 150, 12)
    assert "\n" in w and all(len(line) < len(s) for line in w.split("\n"))
    assert wrap_to_width(s, 5000, 12) == s


def test_figure_draws_and_has_no_overlapping_text(tmp_path):
    out = tmp_path / "fig.png"
    fig = ways_figure(_wav(), 16000, quality=_quality(), quality_hop=0.02, quality_classes=["noise", "reverb", "hum"], models=_models(),
                      window=(0.8, 1.8), title="synthetic", out=out)
    assert out.exists() and out.with_suffix(".svg").exists()
    assert text_overlaps(fig) == [] and fig.ways_overlaps == []
    fig2 = ways_figure(_wav(), 16000, quality=_quality().max(1), quality_hop=0.02, out=tmp_path / "fig2.png")
    assert (tmp_path / "fig2.png").exists() and text_overlaps(fig2) == []


def test_long_title_and_long_row_names_still_do_not_overlap(tmp_path):
    long = "a very long clip title " * 8
    models = [dict(name="a model with an unusually long name that goes on " * 2, tokens=["w%d" % i for i in range(40)],
                   top=[(i % 8, (i * 7) % 30) for i in range(40)], patch_s=0.1, n_bands=8, wer=0.5)]
    fig = ways_figure(_wav(), 16000, quality=_quality(), quality_classes=["noise", "reverb", "hum"], models=models, title=long, out=tmp_path / "long.png")
    assert text_overlaps(fig) == []


def test_text_and_style_are_data(tmp_path):
    out = tmp_path / "custom.png"
    ways_figure(_wav(), 16000, quality=_quality(), quality_classes=["noise", "reverb", "hum"], models=_models(),
                text=dict(title="MY TITLE 123", caption="MY CAPTION 456", y_mel="pitch", x_time="seconds", damage_lane="damage rows"),
                style=dict(width_in=10.0, dpi=100), out=out)
    svg = out.with_suffix(".svg").read_text()
    for s in ("MY TITLE 123", "MY CAPTION 456", "pitch", "seconds", "damage rows"):
        assert s in svg
    with pytest.raises(KeyError):
        ways_figure(_wav(), 16000, models=_models(), text=dict(not_a_field="x"))


def test_same_inputs_same_bytes(tmp_path):
    hashes = []
    for i in range(2):
        out = tmp_path / f"run{i}.png"
        ways_figure(_wav(), 16000, quality=_quality(), quality_classes=["noise", "reverb", "hum"], models=_models(), title="repeat", out=out)
        hashes.append((hashlib.sha256(out.read_bytes()).hexdigest(), hashlib.sha256(out.with_suffix(".svg").read_bytes()).hexdigest()))
    assert hashes[0] == hashes[1]


def test_strict_layout_error_is_raised_when_text_cannot_be_separated(tmp_path):
    # a style whose font is absurdly large cannot be laid out without overlaps
    with pytest.raises(LayoutError):
        ways_figure(_wav(), 16000, quality=_quality(), quality_classes=["noise", "reverb", "hum"], models=_models(), title="x",
                    style=dict(font_size=40.0, label_size=40.0, strip_size=40.0))
    fig = ways_figure(_wav(), 16000, quality=_quality(), quality_classes=["noise", "reverb", "hum"], models=_models(), title="x",
                      style=dict(font_size=40.0, label_size=40.0, strip_size=40.0), strict=False)
    assert fig.ways_overlaps


def test_from_attribution_matches_argmax():
    class Att:
        delta = np.zeros((2, 8, 5), np.float32); pieces = [" a", " b"]; model = "m"; patch_s = 0.1; n_bands = 8
    Att.delta[0, 3, 2] = 1.0
    m = from_attribution(Att())
    assert m["top"] == [(3, 2), None] and m["name"] == "m" and m["tokens"] == [" a", " b"]


def test_latent_figure_from_arrays_no_overlaps_and_editable(tmp_path):
    from ways.latent import degraded_frames, layer_summary, loudness_db
    rng = np.random.default_rng(2)
    wav = _wav(2, 2); n = 100; q = np.zeros((n, 3)); q[30:60, 0] = 0.9
    cam = rng.random((4, n)); shift = rng.random((4, n)) * 0.3; degraded = degraded_frames(q, n)
    summary = layer_summary(cam, shift, degraded, loudness_db(wav, n))
    out = tmp_path / "latent.png"
    fig = latent_figure(wav, 16000, quality=q, quality_classes=["noise", "music", "hum"], cam=cam, shift=shift, degraded=degraded,
                        loudness_db=loudness_db(wav, n), summary=summary, model_name="synthetic",
                        text=LatentText(title="LATENT 789", summary_lane="per layer"), out=out)
    assert out.exists() and text_overlaps(fig) == [] and "LATENT 789" in out.with_suffix(".svg").read_text()


def test_window_accepts_one_span_or_a_list(tmp_path):
    f1 = ways_figure(_wav(), 16000, quality=_quality(), quality_classes=["noise", "reverb", "hum"], models=_models(), window=(0.5, 1.0), out=tmp_path / "w1.png")
    f2 = ways_figure(_wav(), 16000, quality=_quality(), quality_classes=["noise", "reverb", "hum"], models=_models(),
                     window=[(0.3, 0.8), (1.4, 1.9), (2.4, 2.8)], out=tmp_path / "w3.png")
    assert text_overlaps(f1) == [] and text_overlaps(f2) == []
    svg = (tmp_path / "w3.svg").read_text()
    assert svg.count("stroke-dasharray") >= 6          # two dashed edges per window


def test_explicit_line_break_in_title_is_wrapped_per_line_and_nothing_leaves_the_page(tmp_path):
    long = "first line that is long enough to need wrapping on its own " * 3 + "\nsecond line, also long " * 4
    fig = ways_figure(_wav(), 16000, quality=_quality(), quality_classes=["noise", "reverb", "hum"], models=_models(), title=long, out=tmp_path / "nl.png")
    assert text_overlaps(fig) == []


def test_known_windows_are_the_reference_for_the_verdict(tmp_path):
    models = [dict(name="m", tokens=["a", "b", "c"], top=[(1, 5), (2, 20), (3, 27)], patch_s=0.1, n_bands=8)]
    fig = ways_figure(_wav(3), 16000, models=models, window=[(0.4, 0.7), (2.5, 3.0)], out=tmp_path / "win.png")
    svg = (tmp_path / "win.svg").read_text()
    assert "2 of 3 words on the known damaged audio" in svg and text_overlaps(fig) == []
