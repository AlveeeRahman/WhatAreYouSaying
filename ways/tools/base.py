"""The contract every WAYS tool follows.

A tool is one question about one clip. It takes a waveform, a backend and the degradation context,
returns a `ToolResult` (numbers that serialise to JSON, plus the arrays and the figure paths), and
exposes a JSON schema so a language model or a script can call it by name through `ways.tools.call`.
Tools are plain objects; `register` puts them in the registry that the CLI and `schemas()` read.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ToolResult:
    """What a tool returns. `summary` is JSON-serialisable; `data` holds arrays and objects that are not.
    `figure` is the main figure path (None when no `out` was given) and `figures` the named extra ones."""
    tool: str
    summary: dict[str, Any]
    figure: str | None = None
    figures: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(tool=self.tool, summary=self.summary, figure=self.figure, figures=dict(self.figures))

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


@dataclass(frozen=True)
class Tool:
    """A named, documented, schema-carrying callable over one waveform.

    run(wav, sr, **kwargs) -> ToolResult. `parameters` is the JSON schema of the keyword arguments
    (OpenAI function shape; `audio` is added by `schema()` for callers that pass a file path).
    `summary` is one line for `ways tools` and the schema. `needs_white_box` is True when the tool
    reads the encoder, which rules out the openai-compat backend.
    """
    name: str
    summary: str
    run: Callable[..., ToolResult]
    parameters: dict[str, Any]
    needs_white_box: bool = False

    def schema(self) -> dict[str, Any]:
        """The tool as a function-calling schema with `audio` (a file path) as the one required argument."""
        props = {"audio": {"type": "string", "description": "Path to a mono audio file (any sample rate; resampled to 16 kHz)."}}
        props.update(self.parameters.get("properties", {}))
        return {"name": self.name, "description": self.summary,
                "parameters": {"type": "object", "properties": props, "required": ["audio"]}}


COMMON_PARAMETERS: dict[str, Any] = {
    "aadd": {"type": "boolean", "description": "Run AADD v3 on the clip to find degraded audio.", "default": True},
    "window": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
               "description": "Known damaged span [start_s, end_s]; used instead of AADD flags. sound-token also takes a list of spans."},
    "out": {"type": "string", "description": "Output path stem; figures are written next to it."},
    "title": {"type": "string", "description": "Figure title."},
}


def resample_16k(wav, sr: int):
    """Mono float32 at 16 kHz, which every backend and the figure expect."""
    wav = np.asarray(wav, np.float32)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    return wav, 16000


def quality_frames(wav, quality, aadd: bool, aadd_model: str):
    """The [frames, classes] degradation probabilities: the ones given, AADD's, or None."""
    if quality is not None:
        return np.asarray(quality, np.float32)
    if aadd:
        from ..quality import Quality

        return Quality(aadd_model).frames(wav)
    return None
