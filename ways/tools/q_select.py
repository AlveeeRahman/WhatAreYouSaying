"""q-select: several speech models transcribe one clip and a small scorer keeps the transcript it trusts most."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import COMMON_PARAMETERS, Tool, ToolResult, quality_frames, resample_16k
from .registry import register
from .sound_token import backend as load_cached_backend

DEFAULT_BACKENDS = (("hf-whisper", "openai/whisper-tiny"), ("hf-whisper", "openai/whisper-base"), ("hf-moonshine", "UsefulSensors/moonshine-base"))


def q_select(wav, sr=16000, backends=DEFAULT_BACKENDS, scorer="quantum", weights=None, quality=None, aadd=False, aadd_model="v3",
             wer_reference=None, revision=None, **_) -> ToolResult:
    """Transcribe the clip with every backend, score each transcript, and keep the best-scored one.

    `scorer` names a shipped scorer: "quantum" (a four-qubit PennyLane circuit), "quantum-no-entanglement"
    or "classical" (a perceptron, no PennyLane needed). `weights` is a file from `ways.qselect.save_scorer`
    and replaces it. The shipped scorers were fitted for whisper-tiny, whisper-base and moonshine-base
    together, and the disagreement feature needs at least two models. With `aadd` the detector's flagged
    share of the clip is one of the four features. Without it that feature is neutral."""
    from ..metrics import wer
    from ..qselect import GATES, SCORERS, choose, load_scorer
    from ..quality import flagged_mask

    wav, sr = resample_16k(wav, sr)
    if len(backends) < 2:
        raise ValueError("q-select compares models and needs at least two backends")
    if weights is None and scorer not in SCORERS:
        raise ValueError(f"scorer is one of {sorted(SCORERS)}")
    model = load_scorer(Path(weights) if weights else GATES / SCORERS[scorer])
    probs = quality_frames(wav, quality, aadd, aadd_model)
    flagged = None if probs is None else float(flagged_mask(probs).mean())
    names, texts, surprisals = [], [], []
    for kind, mid in backends:
        be = load_cached_backend(kind, mid, revision.get(mid) if isinstance(revision, dict) else revision)
        tr = be.transcribe(wav)
        s = be.surprisal(wav[None], tr)[0][0] if tr.n_tokens else np.zeros(1)
        names.append(mid.split("/")[-1]); texts.append(tr.text); surprisals.append(float(s.mean()))
    best, scores = choose(model, names, texts, surprisals, flagged)
    rows = [dict(model=n, text=t, mean_surprisal=s, score=sc, kept=i == best) for i, (n, t, s, sc) in enumerate(zip(names, texts, surprisals, scores, strict=True))]
    if wer_reference:
        for r in rows:
            r["wer"] = wer(wer_reference, r["text"])["wer"]
    summary = dict(kept=names[best], text=texts[best], scorer=model.gate.config(), flagged_share=flagged, detector_used=probs is not None, transcripts=rows)
    return ToolResult(tool="q-select", summary=summary, figure=None, data=dict(scores=scores))


TOOL = register(Tool(
    name="q-select",
    summary="Several speech models transcribe the clip and a small scorer (a four-qubit circuit or its classical control) keeps the "
            "transcript it predicts to have the fewest errors.",
    run=q_select,
    parameters={"properties": {"aadd": COMMON_PARAMETERS["aadd"],
                "backends": {"type": "array", "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                             "description": "[kind, model id] pairs, at least two. Default: whisper-tiny, whisper-base and moonshine-base.",
                             "default": [list(b) for b in DEFAULT_BACKENDS]},
                "scorer": {"type": "string", "enum": ["quantum", "quantum-no-entanglement", "classical"], "default": "quantum"},
                "weights": {"type": "string", "description": "Scorer weights file from ways.qselect.save_scorer, instead of a shipped scorer."},
                "wer_reference": {"type": "string", "description": "A human transcript, to report each transcript's word error rate."}}},
    needs_white_box=True,
))
