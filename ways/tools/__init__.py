"""The WAYS tools: one question about one clip each, callable from Python, the command line, or a
language model through function calling.

  sound-token   which audio each word used, and whether that audio was clean or degraded
  aud-cam       what degraded audio does inside the encoder, layer by layer

    from ways.tools import sound_token, aud_cam, schemas, call, names
    res = sound_token(wav, sr, backends=[("hf-whisper", "openai/whisper-base")], aadd=True, out="out/clip")
    res.summary, res.figure, res.data["attributions"]
    call("aud-cam", audio="clip.wav", aadd=True, out="out/clip", granular=True).to_dict()

A third-party package adds a tool by registering a `ways.tools.Tool` (see docs/tools.md), either in
code with `ways.tools.register(tool)` or through the `ways.tools` entry-point group.
"""
from __future__ import annotations

import json

from .aud_cam import aud_cam
from .base import Tool, ToolResult
from .registry import get, names, register, schemas, tools
from .sound_token import sound_token

__all__ = ["Tool", "ToolResult", "sound_token", "aud_cam", "get", "names", "register", "schemas", "tools",
           "schema", "schema_json", "call", "TOOLS"]


class _ToolMap(dict):
    """`TOOLS[name]` is the tool's run function; kept for scripts written against the first release."""

    def __missing__(self, name):
        return get(name).run

    def __iter__(self):
        return iter(names())

    def __contains__(self, name):
        return name in names()

    def __len__(self):
        return len(names())


TOOLS = _ToolMap()


def schema() -> list[dict]:
    """JSON schemas for every registered tool, in the OpenAI function shape (for the Anthropic API rename
    `parameters` to `input_schema`)."""
    return schemas()


def schema_json(indent: int = 1) -> str:
    return json.dumps(schema(), indent=indent)


def call(name: str, **kwargs) -> ToolResult:
    """Run a tool by name with the schema's arguments: `audio` is a file path, everything else as in the tool."""
    tool = get(name)
    import soundfile as sf

    audio = kwargs.pop("audio")
    wav, sr = sf.read(audio, dtype="float32")
    if "backends" in kwargs:
        kwargs["backends"] = [tuple(b) for b in kwargs["backends"]]
    if "backend" in kwargs:
        kwargs["backend"] = tuple(kwargs["backend"])
    if kwargs.get("window") is not None:
        w = kwargs["window"]
        kwargs["window"] = [tuple(x) for x in w] if isinstance(w[0], (list, tuple)) else tuple(w)
    return tool.run(wav, sr, **kwargs)
