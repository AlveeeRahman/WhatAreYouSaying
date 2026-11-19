"""The tool registry: the built-in tools, plus any that other packages publish under the
`ways.tools` entry-point group. `ways tools` lists them; `ways <name>` runs them; `schemas()` hands
them to a language model."""
from __future__ import annotations

from .base import Tool

_TOOLS: dict[str, Tool] = {}
_DISCOVERED = False


def register(tool: Tool) -> Tool:
    if tool.name in _TOOLS and _TOOLS[tool.name] is not tool:
        raise ValueError(f"a tool named {tool.name!r} is already registered")
    _TOOLS[tool.name] = tool
    return tool


def _discover():
    """Load third-party tools once: any entry point in the `ways.tools` group that resolves to a Tool."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from importlib.metadata import entry_points

    for ep in entry_points(group="ways.tools"):
        try:
            obj = ep.load()
        except Exception:               # a broken plug-in must not take the CLI down
            continue
        if isinstance(obj, Tool):
            register(obj)


def get(name: str) -> Tool:
    _discover()
    if name not in _TOOLS:
        raise KeyError(f"unknown tool {name!r}; choose one of {sorted(_TOOLS)}")
    return _TOOLS[name]


def names() -> list[str]:
    _discover()
    return list(_TOOLS)


def tools() -> list[Tool]:
    _discover()
    return list(_TOOLS.values())


def schemas() -> list[dict]:
    return [t.schema() for t in tools()]
