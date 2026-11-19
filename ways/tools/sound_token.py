"""sound-token: which audio each word used, and whether that audio was clean or degraded."""
from __future__ import annotations

from pathlib import Path

from ..backends import load_backend
from ..figure import from_attribution, ways_figure
from ..quality import CLASSES, SEG_S
from .base import COMMON_PARAMETERS, Tool, ToolResult, quality_frames, resample_16k
from .registry import register

_BACKENDS: dict = {}


def backend(kind: str, model: str | None = None, revision: str | None = None):
    """One loaded backend per (kind, model, revision) for the life of the process."""
    key = (kind, model, revision)
    if key not in _BACKENDS:
        kw = {} if kind == "openai-compat" else dict(revision=revision)
        _BACKENDS[key] = load_backend(kind, model, **kw)
    return _BACKENDS[key]


def sound_token(wav, sr=16000, backends=(("hf-whisper", "openai/whisper-base"),), quality=None, aadd=False, aadd_model="v3",
                threshold=0.5, window=None, patch_s=0.1, n_bands=8, batch=64, wer_reference=None, out=None, title=None,
                text=None, style=None, revision=None) -> ToolResult:
    """Erasure attribution for one clip through one or more speech-to-text backends, and the per-word figure.

    backends: list of (kind, model id). quality: [frames, 12] AADD probabilities at 20 ms, or aadd=True
    to run AADD here. window: a known damaged span or a list of spans; when given, the figure counts
    words against it instead of the detector. wer_reference: a human transcript, for WER in each row.
    """
    from ..attribution import explain
    from ..metrics import clip_summary, token_summary, wer

    wav, sr = resample_16k(wav, sr)
    probs = quality_frames(wav, quality, aadd, aadd_model)
    models, atts, summaries = [], [], []
    for kind, model in backends:
        be = backend(kind, model, revision.get(model) if isinstance(revision, dict) else revision)
        att = explain(be, wav, patch_s=patch_s, n_bands=n_bands, batch=batch)
        w = wer(wer_reference, att.text)["wer"] if wer_reference else None
        name = model.split("/")[-1] if model else be.name
        models.append(from_attribution(att, name, w)); atts.append(att)
        rows = token_summary(att, probs, threshold)
        summaries.append(dict(clip_summary(att, rows, probs, threshold), name=name, wer=w, text=att.text))
    fig_path = None
    if out is not None:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        fig_path = out if out.suffix else out.with_name(out.name + "__sound-token.png")
        ways_figure(wav, sr, quality=probs, quality_hop=SEG_S, quality_classes=CLASSES if probs is not None and probs.shape[1] == 12 else None,
                    threshold=threshold, models=models, window=window, title=title, out=fig_path, text=text, style=style)
    return ToolResult(tool="sound-token", summary=dict(models=summaries), figure=str(fig_path) if fig_path else None,
                      data=dict(attributions=atts, figure_models=models))


TOOL = register(Tool(
    name="sound-token",
    summary="Which audio each word of a speech-to-text transcript used, by erasing the audio patch by patch, "
            "and whether that audio was clean or degraded. Returns per-word patches and the per-word figure.",
    run=sound_token,
    parameters={"properties": {**COMMON_PARAMETERS,
                "backends": {"type": "array", "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                             "description": "List of [kind, model id], kind in hf-whisper, hf-moonshine, openai-compat.",
                             "default": [["hf-whisper", "openai/whisper-base"]]},
                "wer_reference": {"type": "string", "description": "Human transcript, to report WER."}}},
))
