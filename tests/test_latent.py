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


def test_token_cam_summary_and_figure(tmp_path):
    from ways.figure import token_cam_figure
    from ways.latent import token_cam_summary

    rng = np.random.default_rng(3)
    n, K, L = 150, 4, 3
    degraded = np.zeros(n, bool); degraded[30:80] = True                 # 0.6 to 1.6 s of a 3 s clip
    tcam = np.zeros((K, L, n), np.float32)
    tcam[0, :, 40:50] = 1.0                                              # token 0: sharp, inside the stretch
    tcam[1, :, 100:110] = 1.0                                            # token 1: sharp, outside
    tcam[2, :, :] = 1.0                                                  # token 2: smeared over the clip
    tcam /= np.maximum(tcam.sum(2, keepdims=True), 1e-12)                # token 3 stays empty
    erase = np.zeros((K, 30)); erase[0, 8] = 1; erase[1, 21] = 1; erase[2, 8] = 1; erase[3, 0] = 1
    rows = token_cam_summary(tcam, degraded, ["a", "b", "c", "d"], erase_profile=erase, erase_hop=0.1)
    assert rows[0]["reading"] == "degraded" and rows[0]["cam_in"] == 1.0 and abs(rows[0]["cam_t"] - 0.9) < 1e-6
    assert rows[1]["reading"] == "clean" and rows[1]["cam_in"] == 0.0 and abs(rows[1]["erase_t"] - 2.15) < 1e-6
    assert rows[2]["reading"] == "diffuse" and abs(rows[2]["cam_width_s"] - 3.0) < 1e-4
    assert rows[3]["reading"] == "no CAM" and np.isnan(rows[3]["cam_t"])
    wav = (0.05 * rng.standard_normal(n * 320)).astype(np.float32)
    out = tmp_path / "tokens.png"
    token_cam_figure(wav, 16000, tcam, ["a", "b", "c", "d"], degraded, erase_t=[r.get("erase_t") for r in rows], window=(0.6, 1.6),
                     model_name="synthetic", out=out)
    assert out.exists() and out.with_suffix(".svg").exists()
