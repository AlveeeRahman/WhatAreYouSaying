"""The tools as a callable surface: schemas, the registry, and the granular aud-cam figures drawn from arrays."""
import hashlib
import json

import numpy as np
import pytest

from ways.figure import LatentText, cam_figure, layer_summary_figure, spillover_figure, text_overlaps
from ways.tools import TOOLS, Tool, ToolResult, call, get, names, register, schema, schema_json, schemas


def _wav(seconds=2, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(16000 * seconds) / 16000
    return (0.3 * np.sin(2 * np.pi * 180 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def _latent(n=100, layers=5, seed=4):
    from ways.latent import degraded_frames, layer_summary, loudness_db
    rng = np.random.default_rng(seed)
    q = np.zeros((n, 3), np.float32); q[30:60, 0] = 0.9
    cam = rng.random((layers, n)); shift = rng.random((layers, n)) * 0.3
    wav = _wav(2)
    return wav, q, cam, shift, layer_summary(cam, shift, degraded_frames(q, n), loudness_db(wav, n))


def test_schema_names_match_registry_and_is_json():
    s = schema()
    assert [t["name"] for t in s] == ["sound-token", "aud-cam", "cross-check"] == list(TOOLS)
    for t in s:
        assert t["parameters"]["type"] == "object" and t["parameters"]["required"] == ["audio"]
        assert "audio" in t["parameters"]["properties"]
    assert json.loads(schema_json()) == s
    assert "backends" in s[0]["parameters"]["properties"] and "granular" in s[1]["parameters"]["properties"]


def test_call_rejects_unknown_tool():
    with pytest.raises(KeyError):
        call("not-a-tool", audio="x.wav")


def test_registry_lists_the_three_tools_with_their_contract():
    assert names() == ["sound-token", "aud-cam", "cross-check"] and [t["name"] for t in schemas()] == names()
    assert get("aud-cam").needs_white_box and get("cross-check").needs_white_box and not get("sound-token").needs_white_box
    assert TOOLS["sound-token"] is get("sound-token").run and len(TOOLS) == 3
    with pytest.raises(KeyError):
        get("nope")


def test_a_registered_tool_runs_by_name_and_serialises(tmp_path):
    seen = {}

    def run(wav, sr=16000, aadd=False, out=None, **kw):
        seen.update(n=len(wav), sr=sr, aadd=aadd, kw=kw)
        return ToolResult(tool="echo", summary={"n": len(wav), "x": np.float32(1.5)}, figure=None, data={"wav": wav})

    tool = register(Tool(name="echo", summary="Echo the clip length.", run=run, parameters={"properties": {"aadd": {"type": "boolean"}}}))
    assert "echo" in names() and tool.schema()["parameters"]["required"] == ["audio"]
    import soundfile as sf
    sf.write(tmp_path / "c.wav", _wav(1), 16000)
    res = call("echo", audio=str(tmp_path / "c.wav"), aadd=True, window=[0.1, 0.4])
    assert res.tool == "echo" and seen["n"] == 16000 and seen["aadd"] is True and seen["kw"]["window"] == (0.1, 0.4)
    d = json.loads(res.to_json())
    assert d["summary"] == {"n": 16000, "x": 1.5} and "data" not in d
    with pytest.raises(ValueError):
        register(Tool(name="echo", summary="dup", run=run, parameters={}))


def test_cli_lists_tools_and_schema(capsys):
    from ways.cli import main
    main(["tools"])
    out = capsys.readouterr().out
    assert "sound-token" in out and "aud-cam" in out and "white-box" in out
    main(["tools", "--schema"])
    assert [t["name"] for t in json.loads(capsys.readouterr().out)][:2] == ["sound-token", "aud-cam"]


def test_granular_figures_draw_without_overlaps_and_are_editable(tmp_path):
    wav, q, cam, shift, summary = _latent()
    f1 = cam_figure(wav, 16000, cam, quality=q, model_name="synthetic", out=tmp_path / "cam.png")
    f2 = spillover_figure(wav, 16000, shift, quality=q, model_name="synthetic", out=tmp_path / "spill.png")
    f3 = layer_summary_figure(summary, model_name="synthetic", out=tmp_path / "layers.png", text=LatentText(title="LAYERS 321", x_layer="depth"))
    for f in (f1, f2, f3):
        assert text_overlaps(f) == [] and f.ways_overlaps == []
    for n in ("cam", "spill", "layers"):
        assert (tmp_path / f"{n}.png").exists() and (tmp_path / f"{n}.svg").exists()
    svg = (tmp_path / "layers.svg").read_text()
    assert "LAYERS 321" in svg and "depth" in svg
    assert "synthetic: Grad-CAM per encoder layer" in (tmp_path / "cam.svg").read_text()


def test_granular_figures_same_inputs_same_bytes(tmp_path):
    wav, q, cam, shift, summary = _latent()
    h = []
    for i in range(2):
        spillover_figure(wav, 16000, shift, quality=q, model_name="m", out=tmp_path / f"s{i}.png")
        h.append(hashlib.sha256((tmp_path / f"s{i}.png").read_bytes()).hexdigest())
    assert h[0] == h[1]


def test_granular_figures_with_many_layers_and_long_title(tmp_path):
    wav, q, cam, shift, summary = _latent(layers=24)
    f = cam_figure(wav, 16000, cam, quality=q, title="a long title " * 12, out=tmp_path / "deep.png")
    assert text_overlaps(f) == []


def test_aud_cam_schema_has_tokens_flag():
    from ways.tools import get

    props = get("aud-cam").parameters["properties"]
    assert props["tokens"]["type"] == "boolean" and props["tokens"]["default"] is False


def test_cross_check_figure_and_table_from_arrays(tmp_path):
    from ways.figure import cross_check_figure
    from ways.tools.cross_check import table

    rng = np.random.default_rng(5)
    wav = _wav(3); n, K, L = 150, 4, 3
    degraded = np.zeros(n, bool); degraded[30:80] = True
    tcam = rng.random((K, L, n)).astype(np.float32); tcam[0, :, :30] = 0; tcam[0, :, 80:] = 0
    tcam /= tcam.sum(2, keepdims=True)
    model = dict(name="synthetic", tokens=[" a", " b", " c", "."], top=[(1, 10), (3, 20), (5, 24), None], patch_s=0.1, n_bands=8, wer=0.25)
    on = [True, False, False, False]
    out = tmp_path / "x.png"
    cross_check_figure(wav, 16000, model, on, tcam, degraded, [1.05, 2.05, 2.45, None], window=(0.6, 1.6),
                       counts=dict(n_erase=1, n_grad=1, n_both=1), out=out)
    assert out.exists()
    res = ToolResult(tool="cross-check", summary=dict(tokens=[dict(k=0, piece=" a", erase_t=1.05, erase_band=1, erase_in=0.9, cam_t=1.1, cam_width_s=0.4,
                                                                 cam_in=0.95, cam_in_min_layers=0.8, reading="degraded", on_degraded_erasure=True,
                                                                 on_degraded_gradient=True),
                                                            dict(k=1, piece=" b", erase_t=2.05, erase_band=3, erase_in=0.1, cam_t=2.0, cam_width_s=0.5,
                                                                 cam_in=0.05, cam_in_min_layers=0.0, reading="clean", on_degraded_erasure=False,
                                                                 on_degraded_gradient=False)]))
    t = table(res); t2 = table(res, only_degraded=True)
    assert t.count("\n") == 4 and t2.count("\n") == 3 and "| 0 | a | 1.05 | 1 | 0.90 |" in t
