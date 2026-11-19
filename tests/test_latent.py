"""The latent-view bookkeeping on synthetic arrays, CPU only: degraded frames, erasure of a stretch, the
per-layer summary and the figure."""
import numpy as np

from ways.figure import latent_figure
from ways.latent import degraded_frames, erase_frames, layer_summary, loudness_db


def test_degraded_frames_from_window_and_from_flags():
    d = degraded_frames(None, 100, window=(0.5, 1.0))
    assert d[25:50].all() and not d[:25].any() and not d[50:].any()
    q = np.zeros((100, 2)); q[10:20, 1] = 0.9
    d = degraded_frames(q, 100)
    assert d.sum() == 10 and d[10:20].all()


def test_erase_frames_changes_only_the_stretch():
    rng = np.random.default_rng(0)
    wav = (0.1 * rng.standard_normal(16000 * 2)).astype(np.float32)
    d = degraded_frames(None, 100, window=(0.5, 1.0))
    e = erase_frames(wav, d)
    assert e.shape == wav.shape
    assert np.abs(e[:int(0.45 * 16000)] - wav[:int(0.45 * 16000)]).max() < 1e-4      # before the stretch, untouched
    inside = slice(int(0.55 * 16000), int(0.95 * 16000))
    assert (e[inside] ** 2).mean() < 0.05 * (wav[inside] ** 2).mean()                 # inside, mostly gone
    assert np.array_equal(erase_frames(wav, np.zeros(100, bool)), wav)                # nothing to erase, identity


def test_layer_summary_reads_as_designed():
    T = 100; degraded = np.zeros(T, bool); degraded[:50] = True
    cam = np.ones((2, T)); cam[1, :50] = 3.0                    # layer 1 leans on the degraded half
    shift = np.zeros((2, T)); shift[:, :50] = 0.5; shift[1, 50:] = 0.2
    loud = np.linspace(-40, -10, T)
    s = layer_summary(cam, shift, degraded, loud)
    assert abs(s[0]["cam_focus"] - 1.0) < 1e-9 and abs(s[1]["cam_focus"] - 1.5) < 1e-9
    assert s[0]["shift_out"] == 0.0 and abs(s[1]["shift_out"] - 0.2) < 1e-9 and abs(s[1]["shift_in"] - 0.5) < 1e-9
    assert s[1]["rho_loud"] < 0                                 # CAM is high where it is quiet


def test_latent_figure_from_arrays(tmp_path):
    rng = np.random.default_rng(2)
    wav = (0.2 * np.sin(2 * np.pi * 180 * np.arange(32000) / 16000) + 0.02 * rng.standard_normal(32000)).astype(np.float32)
    n = 100; q = np.zeros((n, 3)); q[30:60, 0] = 0.9
    cam = rng.random((4, n)); shift = rng.random((4, n)) * 0.3; degraded = degraded_frames(q, n)
    summary = layer_summary(cam, shift, degraded, loudness_db(wav, n))
    out = tmp_path / "latent.png"
    latent_figure(wav, 16000, quality=q, quality_classes=["noise", "music", "hum"], cam=cam, shift=shift, degraded=degraded,
                  loudness_db=loudness_db(wav, n), summary=summary, model_name="synthetic", title="t", out=out)
    assert out.exists() and out.with_suffix(".svg").exists()
