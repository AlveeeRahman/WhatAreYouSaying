# Help wanted

WAYS is at version 0.3.2 and open to contributors. These are the open pieces of work, each sized so one person can own it. Read [CONTRIBUTING.md](../CONTRIBUTING.md) first, and open an issue before starting anything bigger than a fix so we can agree on the approach.

## Good first issues

### A CI workflow

The commands are fixed: `make test` (the CPU suite, about half a minute) and `make lint` (ruff). A GitHub Actions workflow should run both on pull requests, on Linux, with CPU torch (`make install` uses CPU torch by default). The two end-to-end HTTP checks need a GPU and stay out (`-m "gpu or slow"` selects them). When it passes, the README's "CI: not set up yet" badge can point at the workflow.

### Reproduce a result on your machine

Run `pytest tests -q`, then one clip through `ways explain` or `ways sound-token`, and report what you ran it on. A table in the README only changes from a run that wrote a `provenance.json`, so attach that file. CPU-only runs are useful too: they tell us what a first-time user sees.

## Bigger pieces

### A live vLLM or llama.cpp run

The `openai-compat` backend has only run against `tests/mock_server.py`. Point it at a real server, for example:

```sh
vllm serve openai/whisper-large-v3 --port 8000
ways explain clip.wav --backend openai-compat --route transcriptions --base-url http://localhost:8000/v1 \
     --model openai/whisper-large-v3 --patch 0.2 --bands 4 --out out/
```

Report what worked, what broke, and how long a clip took. `docs/backends.md` has the request shapes both servers document.

### A larger model on the same clips

Every result so far comes from whisper-tiny, whisper-base and moonshine-base. Running a larger model on the LibriSpeech controlled set or the VoiceBank-DEMAND set (both CC BY 4.0, built by `experiments/controlled.py` and `experiments/demand.py`) would show whether the findings hold at scale.

### Other languages

Every dataset so far is English. A non-English test set with known degradations, run through a multilingual Whisper model, would test whether the evidence maps behave the same way.

### A new tool

A tool is one question about one clip: a `run(wav, sr, **kwargs) -> ToolResult` function with its keyword arguments described as a JSON schema, registered in `ways.tools` or published under the `ways.tools` entry-point group from your own package. `docs/tools.md` and CONTRIBUTING.md show how. Ideas: a per-speaker view for multi-speaker clips, or a comparison of two models' maps on the same clip.

## How to get in touch

Comment on the matching issue, or open one. Security problems go through the private route in [SECURITY.md](../SECURITY.md).
