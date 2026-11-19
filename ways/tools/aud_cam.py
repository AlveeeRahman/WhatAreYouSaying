"""aud-cam: what degraded audio does inside the encoder, layer by layer (spill-over under erasure and Grad-CAM)."""
from __future__ import annotations

from pathlib import Path

from ..figure import cam_figure, latent_figure, layer_summary_figure, spillover_figure, token_cam_figure
from ..quality import CLASSES, SEG_S
from .base import COMMON_PARAMETERS, Tool, ToolResult, quality_frames, resample_16k
from .registry import register
from .sound_token import backend as load_cached_backend


def aud_cam(wav, sr=16000, backend=("hf-whisper", "openai/whisper-base"), quality=None, aadd=False, aadd_model="v3",
            threshold=0.5, window=None, out=None, granular=False, tokens=False, erase_t=None, title=None, text=None, style=None,
            revision=None) -> ToolResult:
    """Spill-over and Grad-CAM per encoder layer for one clip, the latent figure, and with granular=True
    the Grad-CAM lane, the spill-over lane and the per-layer summary as separate figures.

    backend: (kind, model id) of a white-box backend (hf-whisper or hf-moonshine). window: (start_s, end_s)
    of the stretch to erase instead of AADD's flags. tokens=True adds a per-token saliency map (gradient times
    activation per frame, one backward pass per token; see ways.latent.token_cam), its per-token summary and
    the token figure; erase_t, a time per token from the erasure test, is drawn on that figure when given.
    """
    from ..latent import latent_view, token_cam, token_cam_summary

    wav, sr = resample_16k(wav, sr)
    kind, model = tuple(backend)
    be = load_cached_backend(kind, model, revision)
    probs = quality_frames(wav, quality, aadd, aadd_model)
    tr = be.transcribe(wav)
    r = latent_view(be, wav, probs=probs, window=window, threshold=threshold, transcript=tr)
    name = model.split("/")[-1] if model else be.name
    tcam, per_token = None, None
    if tokens:
        tcam, _ = token_cam(be, wav, tr)
        per_token = token_cam_summary(tcam, r.degraded, tr.pieces)
    classes = CLASSES if probs is not None and probs.shape[1] == 12 else None
    figs, fig_path = {}, None
    if out is not None:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.with_suffix("") if out.suffix else out
        fig_path = stem.with_name(stem.name + "__aud-cam.png")
        latent_figure(wav, sr, quality=probs, quality_hop=SEG_S, quality_classes=classes, threshold=threshold, cam=r.cam, shift=r.shift,
                      degraded=r.degraded, loudness_db=r.loudness_db, summary=r.summary, model_name=name, title=title, out=fig_path,
                      text=text, style=style)
        if granular:
            figs["cam"] = str(stem.with_name(stem.name + "__aud-cam-cam.png"))
            cam_figure(wav, sr, r.cam, quality=probs, quality_hop=SEG_S, threshold=threshold, model_name=name, out=figs["cam"], text=text, style=style)
            figs["spillover"] = str(stem.with_name(stem.name + "__aud-cam-spillover.png"))
            spillover_figure(wav, sr, r.shift, quality=probs, quality_hop=SEG_S, threshold=threshold, model_name=name, out=figs["spillover"],
                             text=text, style=style)
            figs["layers"] = str(stem.with_name(stem.name + "__aud-cam-layers.png"))
            layer_summary_figure(r.summary, model_name=name, out=figs["layers"], text=text, style=style)
        if tokens:
            figs["tokens"] = str(stem.with_name(stem.name + "__aud-cam-tokens.png"))
            token_cam_figure(wav, sr, tcam, tr.pieces, r.degraded, erase_t=erase_t, window=window, model_name=name, out=figs["tokens"],
                             text=text, style=style)
    summary = dict(model=name, text=tr.text, layers=r.layers, degraded_share=float(r.degraded.mean()), per_layer=r.summary)
    if per_token is not None:
        summary["per_token"] = per_token
    return ToolResult(tool="aud-cam", summary=summary, figure=str(fig_path) if fig_path else None, figures=figs,
                      data=dict(result=r, token_cam=tcam, transcript=tr))


TOOL = register(Tool(
    name="aud-cam",
    summary="What degraded audio does to a speech-to-text model's encoder, layer by layer: spill-over of the "
            "representation under erasure of the degraded stretch, Grad-CAM per layer and, on request, a gradient saliency per token. "
            "Returns per-layer (and per-token) numbers and figures.",
    run=aud_cam,
    parameters={"properties": {**COMMON_PARAMETERS,
                "backend": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2,
                            "description": "[kind, model id], kind in hf-whisper, hf-moonshine.", "default": ["hf-whisper", "openai/whisper-base"]},
                "granular": {"type": "boolean", "default": False,
                             "description": "Also write the Grad-CAM lane, the spill-over lane and the per-layer summary as separate figures."},
                "tokens": {"type": "boolean", "default": False,
                           "description": "Also compute a gradient saliency per token (one backward pass each), with a per-token summary and figure."}}},
    needs_white_box=True,
))
