# The tool framework

A WAYS tool is one question about one clip. It takes a waveform and a sample rate, a backend and
the degradation context, and returns a `ToolResult`. Two ship with the package:

| tool | question | backends | main figure |
|---|---|---|---|
| `sound-token` | which audio did each word use, and was that audio clean or degraded | any (`openai-compat` gives the black-box word-change map) | the per-word figure (`ways_figure`) |
| `aud-cam` | what does the degraded audio do inside the encoder, layer by layer | white-box only (`hf-whisper`, `hf-moonshine`) | the latent figure (`latent_figure`), three granular lanes with `granular=True`, and a per-token gradient saliency with figure and summary with `tokens=True` |

## The contract

```python
from ways.tools import Tool, ToolResult, register

@dataclass(frozen=True)
class Tool:
    name: str                 # the subcommand and the function-calling name
    summary: str              # one line, shown by `ways tools` and used as the schema description
    run: Callable[..., ToolResult]   # run(wav, sr, **kwargs)
    parameters: dict          # JSON schema of the keyword arguments: {"properties": {...}}
    needs_white_box: bool = False

@dataclass
class ToolResult:
    tool: str
    summary: dict             # JSON-serialisable numbers and strings
    figure: str | None        # the main figure's path, when `out` was given
    figures: dict[str, str]   # named extra figures
    data: dict                # arrays and objects that are not JSON (attributions, LatentResult)
```

`Tool.schema()` adds `audio` (a file path) as the one required argument and returns the OpenAI
function shape; for the Anthropic API rename `parameters` to `input_schema`.

## Calling

```python
from ways.tools import sound_token, aud_cam, call, schemas, names

res = sound_token(wav, 16000, backends=[("hf-whisper", "openai/whisper-base")], aadd=True, out="out/clip")
res.summary["models"][0]["focus_ratio_mean"]; res.figure; res.data["attributions"][0]

res = aud_cam(wav, 16000, backend=("hf-moonshine", "UsefulSensors/moonshine-base"), window=(1.5, 3.8), out="out/clip", granular=True, tokens=True)
res.summary["per_layer"][-1]["shift_out"]; res.figures["spillover"]

call("sound-token", audio="clip.wav", aadd=True, out="out/clip").to_json()   # by name, from a file
schemas()                                                                    # hand these to a language model
```

From the command line every registered tool is a subcommand with the same flags:

```sh
ways tools                       # the registry, one line per tool; --schema prints the JSON schemas
ways sound-token clip.wav --backend hf-whisper=openai/whisper-base --backend hf-moonshine=UsefulSensors/moonshine-base --aadd --out out/clip
ways aud-cam clip.wav --backend hf-whisper=openai/whisper-base --window 1.5 3.8 --granular --tokens --out out/clip
```

## Writing a tool

1. Write `run(wav, sr, **kwargs) -> ToolResult`. Use `ways.tools.base.resample_16k` and
   `quality_frames` for the inputs every tool shares, and `COMMON_PARAMETERS` for their schema.
2. Draw through `ways/figure.py`, so every string is data, no text overlaps, and the same inputs
   give the same bytes. New figure kinds get their own templated function there.
3. Register it. Inside this repository: a module under `ways/tools/`, imported from
   `ways/tools/__init__.py`. From another package: an entry point in the `ways.tools` group that
   resolves to the `Tool` instance.
4. Test it from synthetic arrays on the CPU (`tests/test_tools.py` shows the pattern), and run it
   once on a real clip before claiming a figure.

Backends are loaded once per process and shared between tools (`ways.tools.sound_token.backend`).
