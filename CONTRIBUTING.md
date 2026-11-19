# Contributing

Thanks for helping. This page covers how to set up, test and send a change, and how to add a tool.

## Set up and test

You need Python 3.12 and a virtual environment. A GPU is optional for the tests.

```sh
python -m venv .venv && . .venv/bin/activate
make install                 # pinned requirements (CPU torch by default; TORCH_INDEX=cu128 for CUDA) and the package
make test                    # the CPU suite, about half a minute
make lint                    # ruff
make clean-env               # the clean-environment execution test (fresh venv, pip check, tests, lock file)
```

`pytest tests -q -m "gpu or slow"` runs the two end-to-end checks against the mock
OpenAI-compatible server; they need a CUDA device, the cached whisper-tiny weights and one clip
from the data folder (`WAYS_DATA`, layout in `experiments/common.py`).

## Adding a tool

A tool is one question about one clip. Write a function `run(wav, sr, **kwargs) -> ToolResult`,
describe its keyword arguments as a JSON schema, and register it:

```python
from ways.tools import Tool, ToolResult, register

def my_tool(wav, sr=16000, aadd=False, out=None, **_):
    ...
    return ToolResult(tool="my-tool", summary={"answer": 1.0}, figure=None)

register(Tool(name="my-tool", summary="One line saying what it answers.", run=my_tool,
              parameters={"properties": {"aadd": {"type": "boolean"}, "out": {"type": "string"}}}))
```

Inside this repository, add the module under `ways/tools/` and import it from
`ways/tools/__init__.py`. From another package, publish the `Tool` under the `ways.tools`
entry-point group and it appears in `ways tools` and runs as `ways my-tool`. Figures go through
`ways/figure.py`: every string a field of a text dataclass, the overlap and page-edge check on, the
same bytes for the same inputs. `docs/tools.md` has the details.

## Sending a change

1. Open an issue first for anything bigger than a fix, so we can agree on the approach.
2. Keep a pull request to one change. Update the docs and `CHANGELOG.md` in the same pull request.
3. Run `make test` and `make lint`. Continuous integration is not set up yet; the commands a
   workflow would run are these two.
4. Say what you ran on real hardware, if anything. A result table in the README comes from a run
   that wrote a `provenance.json`; a figure comes from the script named next to it.

## Style

Python 3.12, type hints where they help, docstrings that say what a function answers. No absolute
paths, user names or machine layouts in any tracked file: data comes from `WAYS_DATA`, the
interpreter from `python` on the path. Prose in the README and the docs is plain: short sentences,
no em dashes, numbers with their source.

## Reporting a security problem

Use the private route in [SECURITY.md](SECURITY.md), not a public issue.
