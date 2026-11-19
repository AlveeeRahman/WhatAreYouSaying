"""The two WAYS tools, callable from Python, from the command line, or by an LLM through function calling.

  sound-token   which audio each word used: erasure attribution per token, drawn with ways_figure
                (damage rows, spectrogram with the degraded spans and one box per word, the
                transcript with words on degraded audio underlined, the count against chance).
  aud-cam       what degraded audio does to the model's latent space: spill-over per encoder layer
                under erasure of the degraded stretch, plus Grad-CAM per layer, drawn with
                latent_figure; with granular=True the Grad-CAM lane, the spill-over lane and the
                per-layer summary are also written as separate figures.

Both take a waveform, a backend (hf-whisper or hf-moonshine; aud-cam needs a white-box backend),
optional AADD scores or a known damaged window, and optional text and style overrides for the
figure. `schema()` returns JSON schemas for the two tools, `call(name, **kwargs)` runs one by name.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .figure import cam_figure, from_attribution, latent_figure, layer_summary_figure, spillover_figure, ways_figure
from .quality import CLASSES, SEG_S

_BACKENDS = {}


def _backend(kind, model=None, revision=None):
    from .backends import load_backend

    key = (kind, model, revision)
    if key not in _BACKENDS:
        kw = {} if kind == "openai-compat" else dict(revision=revision)
        _BACKENDS[key] = load_backend(kind, model, **kw)
    return _BACKENDS[key]


def _quality(wav, quality, aadd, aadd_model):
    if quality is not None:
        return np.asarray(quality, np.float32)
    if aadd:
        from .quality import Quality

        return Quality(aadd_model).frames(wav)
    return None


def sound_token(wav, sr=16000, backends=(("hf-whisper", "openai/whisper-base"),), quality=None, aadd=False, aadd_model="v3",
                threshold=0.5, window=None, patch_s=0.1, n_bands=8, batch=64, wer_reference=None, out=None, title=None,
                text=None, style=None, revision=None):
    """Erasure attribution for one clip through one or more speech-to-text backends, and the figure.

    backends: list of (kind, model id). quality: [frames, 12] AADD probabilities at 20 ms, or aadd=True
    to run AADD here. wer_reference: a human transcript, to put WER in each row title.
    Returns dict(models=[figure entries], attributions=[Attribution], summaries=[clip summaries], figure=path or None).
    """
    from . import explain
    from .metrics import clip_summary, token_summary, wer

    wav = np.asarray(wav, np.float32)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32); sr = 16000
    probs = _quality(wav, quality, aadd, aadd_model)
    models, atts, summaries = [], [], []
    for kind, model in backends:
        be = _backend(kind, model, revision)
        att = explain(be, wav, patch_s=patch_s, n_bands=n_bands, batch=batch)
        w = wer(wer_reference, att.text)["wer"] if wer_reference else None
        name = model.split("/")[-1] if model else be.name
        models.append(from_attribution(att, name, w)); atts.append(att)
        rows = token_summary(att, probs, threshold); summaries.append(dict(clip_summary(att, rows, probs, threshold), name=name, wer=w, text=att.text))
    fig_path = None
    if out is not None:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        fig_path = out if out.suffix else out.with_name(out.name + "__sound-token.png")
        ways_figure(wav, sr, quality=probs, quality_hop=SEG_S, quality_classes=CLASSES if probs is not None and probs.shape[1] == 12 else None,
                    threshold=threshold, models=models, window=window, title=title, out=fig_path, text=text, style=style)
    return dict(models=models, attributions=atts, summaries=summaries, figure=str(fig_path) if fig_path else None)


def aud_cam(wav, sr=16000, backend=("hf-whisper", "openai/whisper-base"), quality=None, aadd=False, aadd_model="v3", threshold=0.5,
            window=None, out=None, granular=False, title=None, text=None, style=None, revision=None):
    """Spill-over and Grad-CAM per encoder layer for one clip, and the figure(s).

    Returns dict(result=LatentResult, summary=[per layer], figure=path, figures={lane: path} when granular).
    """
    from .latent import latent_view

    wav = np.asarray(wav, np.float32)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32); sr = 16000
    kind, model = backend
    be = _backend(kind, model, revision)
    probs = _quality(wav, quality, aadd, aadd_model)
    r = latent_view(be, wav, probs=probs, window=window, threshold=threshold)
    name = model.split("/")[-1] if model else be.name
    classes = CLASSES if probs is not None and probs.shape[1] == 12 else None
    figs, fig_path = {}, None
    if out is not None:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.with_suffix("") if out.suffix else out
        fig_path = stem.with_name(stem.name + "__aud-cam.png")
        latent_figure(wav, sr, quality=probs, quality_hop=SEG_S, quality_classes=classes, threshold=threshold, cam=r.cam, shift=r.shift,
                      degraded=r.degraded, loudness_db=r.loudness_db, summary=r.summary, model_name=name, title=title, out=fig_path, text=text, style=style)
        if granular:
            figs["cam"] = str(stem.with_name(stem.name + "__aud-cam-cam.png"))
            cam_figure(wav, sr, r.cam, quality=probs, quality_hop=SEG_S, threshold=threshold, model_name=name, out=figs["cam"], text=text, style=style)
            figs["spillover"] = str(stem.with_name(stem.name + "__aud-cam-spillover.png"))
            spillover_figure(wav, sr, r.shift, quality=probs, quality_hop=SEG_S, threshold=threshold, model_name=name, out=figs["spillover"], text=text, style=style)
            figs["layers"] = str(stem.with_name(stem.name + "__aud-cam-layers.png"))
            layer_summary_figure(r.summary, model_name=name, out=figs["layers"], text=text, style=style)
    return dict(result=r, summary=r.summary, figure=str(fig_path) if fig_path else None, figures=figs)


TOOLS = {"sound-token": sound_token, "aud-cam": aud_cam}


def schema():
    """JSON schemas for the two tools, in the shape function-calling APIs take."""
    common = {
        "audio": {"type": "string", "description": "Path to a mono audio file (any sample rate; resampled to 16 kHz)."},
        "aadd": {"type": "boolean", "description": "Run AADD v3 on the clip to find degraded audio.", "default": True},
        "window": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                   "description": "Known damaged span [start_s, end_s]; used instead of AADD flags. sound-token also takes a list of spans."},
        "out": {"type": "string", "description": "Output path stem; figures are written next to it."},
        "title": {"type": "string", "description": "Figure title."},
    }
    return [
        {"name": "sound-token",
         "description": "Which audio each word of a speech-to-text transcript used, by erasing the audio patch by patch, "
                        "and whether that audio was clean or degraded. Returns per-word patches and a figure.",
         "parameters": {"type": "object", "properties": {**common,
                        "backends": {"type": "array", "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                                     "description": "List of [kind, model id], kind in hf-whisper, hf-moonshine, openai-compat.",
                                     "default": [["hf-whisper", "openai/whisper-base"]]},
                        "wer_reference": {"type": "string", "description": "Human transcript, to report WER."}},
                        "required": ["audio"]}},
        {"name": "aud-cam",
         "description": "What degraded audio does to a speech-to-text model's encoder, layer by layer: spill-over of the "
                        "representation under erasure of the degraded stretch, and Grad-CAM per layer. Returns per-layer numbers and figures.",
         "parameters": {"type": "object", "properties": {**common,
                        "backend": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2,
                                    "description": "[kind, model id], kind in hf-whisper, hf-moonshine.", "default": ["hf-whisper", "openai/whisper-base"]},
                        "granular": {"type": "boolean", "description": "Also write the Grad-CAM lane, the spill-over lane and the per-layer summary as separate figures.", "default": False}},
                        "required": ["audio"]}},
    ]


def call(name, **kwargs):
    """Run a tool by name with the schema's arguments (audio path instead of a waveform)."""
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name!r}; choose one of {sorted(TOOLS)}")
    import soundfile as sf

    audio = kwargs.pop("audio")
    wav, sr = sf.read(audio, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if "backends" in kwargs:
        kwargs["backends"] = [tuple(b) for b in kwargs["backends"]]
    if "backend" in kwargs:
        kwargs["backend"] = tuple(kwargs["backend"])
    if kwargs.get("window") is not None:
        w = kwargs["window"]
        kwargs["window"] = [tuple(x) for x in w] if isinstance(w[0], (list, tuple)) else tuple(w)
    out = TOOLS[name](wav, sr, **kwargs)
    return {k: v for k, v in out.items() if k not in ("attributions", "result")}


def schema_json(indent=1):
    return json.dumps(schema(), indent=indent)
