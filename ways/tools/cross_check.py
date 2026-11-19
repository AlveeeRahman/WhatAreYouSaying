"""cross-check: the tokens on degraded audio under both readings, the erasure test (sound-token) and the
encoder's gradient map (aud-cam), token by token, and whether the two agree."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..figure import cross_check_figure, from_attribution
from ..quality import CLASSES, SEG_S
from .base import COMMON_PARAMETERS, Tool, ToolResult, quality_frames, resample_16k
from .registry import register
from .sound_token import backend as load_cached_backend


def cross_check(wav, sr=16000, backend=("hf-whisper", "openai/whisper-base"), quality=None, aadd=False, aadd_model="v3", threshold=0.5,
                window=None, patch_s=0.1, n_bands=8, batch=64, wer_reference=None, out=None, title=None, text=None, style=None,
                revision=None) -> ToolResult:
    """One model, one clip, one transcript, two readings of every token: the erasure test (which 100 ms x band
    patch mattered most, and how much of the token's erasure mass lies on the degraded audio) and the encoder's
    per-frame gradient map (where it sits, how wide, how much of it lies on the degraded audio at the top layer
    and at every layer). The summary lists every token with both readings and a verdict (degraded, clean,
    split, diffuse), and counts the tokens on degraded audio by each method and by both.

    backend: (kind, model id) of a white-box backend. window: (start_s, end_s) of a known damaged span, or
    aadd=True / quality for the detector's flags. The figure stacks the erasure boxes, the gradient lanes and
    the per-layer shares; `table(result)` renders the summary as Markdown.
    """
    from ..attribution import explain
    from ..latent import degraded_frames, token_cam, token_cam_summary
    from ..metrics import token_mass, wer

    wav, sr = resample_16k(wav, sr)
    kind, model = tuple(backend)
    be = load_cached_backend(kind, model, revision)
    if not getattr(be, "supports_surprisal", False):
        raise NotImplementedError(f"{be.name} is a black-box backend: cross-check needs the encoder")
    probs = quality_frames(wav, quality, aadd, aadd_model)
    tr = be.transcribe(wav)
    att = explain(be, wav, patch_s=patch_s, n_bands=n_bands, batch=batch, transcript=tr)
    n20 = int(len(wav) // (SEG_S * 16000))
    degraded = degraded_frames(probs, n20, threshold, window)
    tcam, _ = token_cam(be, wav, tr)
    w = wer(wer_reference, att.text)["wer"] if wer_reference else None
    name = model.split("/")[-1] if model else be.name
    entry = from_attribution(att, name, w)                   # the same first-ranked (band, patch) as the sound-token figure draws
    m = token_mass(att)
    prof = np.where(np.isnan(m).all((1, 2))[:, None], np.nan, m.sum(1))
    erase_t = [float((tp[1] + 0.5) * patch_s) if tp is not None else float("nan") for tp in entry["top"]]
    rows = token_cam_summary(tcam, degraded, tr.pieces, erase_profile=prof, erase_hop=patch_s)
    on = []
    for r, t, tp in zip(rows, erase_t, entry["top"], strict=True):
        r["erase_t"] = t
        r["erase_band"] = int(tp[0]) if tp is not None else None
        r["cam_in_min_layers"] = float(np.nanmin(r["cam_in_layers"])) if np.isfinite(r["cam_in_layers"]).any() else float("nan")
        r["on_degraded_erasure"] = bool(degraded[min(int(t / SEG_S), n20 - 1)]) if np.isfinite(t) else False
        r["on_degraded_gradient"] = bool(np.isfinite(r["cam_in"]) and r["cam_in"] >= 0.5)
        on.append(r["on_degraded_erasure"])
    counts = dict(n_erase=int(sum(r["on_degraded_erasure"] for r in rows)), n_grad=int(sum(r["on_degraded_gradient"] for r in rows)),
                  n_both=int(sum(r["on_degraded_erasure"] and r["on_degraded_gradient"] for r in rows)))
    fig_path = None
    if out is not None:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.with_suffix("") if out.suffix else out
        fig_path = stem.with_name(stem.name + "__cross-check.png")
        cross_check_figure(wav, sr, entry, on, tcam, degraded, erase_t, quality=probs, quality_hop=SEG_S,
                           quality_classes=CLASSES if probs is not None and probs.shape[1] == 12 else None, threshold=threshold, window=window,
                           counts=counts, title=title, out=fig_path, text=text, style=style)
    summary = dict(model=name, text=tr.text, wer=w, n_tokens=len(rows), degraded_share=float(degraded.mean()), on_degraded=counts, tokens=rows)
    return ToolResult(tool="cross-check", summary=summary, figure=str(fig_path) if fig_path else None,
                      data=dict(attribution=att, token_cam=tcam, degraded=degraded, transcript=tr, figure_model=entry))


def table(result: ToolResult, only_degraded: bool = False) -> str:
    """The per-token summary as a Markdown table: one row per token (or only the tokens either reading puts on
    degraded audio), the erasure reading on the left, the encoder reading on the right."""
    head = ("| k | token | erasure patch (s) | band | erasure share on degraded | gradient centre (s) | width (s) | "
            "gradient share, top layer | lowest share over layers | reading |")
    lines = [head, "|" + "---|" * 10]
    for r in result.summary["tokens"]:
        if only_degraded and not (r["on_degraded_erasure"] or r["on_degraded_gradient"]):
            continue
        f = lambda v, d=2: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"   # noqa: E731
        lines.append(f"| {r['k']} | {r['piece'].strip() or '_'} | {f(r.get('erase_t'))} | {r.get('erase_band', '-')} | {f(r.get('erase_in'))} | "
                     f"{f(r['cam_t'])} | {f(r['cam_width_s'])} | {f(r['cam_in'])} | {f(r['cam_in_min_layers'])} | {r['reading']} |")
    return "\n".join(lines) + "\n"


TOOL = register(Tool(
    name="cross-check",
    summary="The tokens on degraded audio under both readings, the erasure test (sound-token) and the encoder's gradient map "
            "(aud-cam), token by token, with a verdict per token and the counts by each method and by both.",
    run=cross_check,
    parameters={"properties": {**COMMON_PARAMETERS,
                "backend": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2,
                            "description": "[kind, model id], kind in hf-whisper, hf-moonshine.", "default": ["hf-whisper", "openai/whisper-base"]},
                "wer_reference": {"type": "string", "description": "A human transcript, to report the word error rate."}}},
    needs_white_box=True,
))
