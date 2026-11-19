# Changelog

## Unreleased

- **docs/help-wanted.md**: the open work for contributors (CI workflow, a live vLLM or llama.cpp run, a
  larger model, other languages, new tools, reproduction reports), linked from the README.

## 0.3.2

- **Two checks of the method.** `experiments/check_timestamps.py` joins pairs of utterances and
  confirms that a Whisper token timestamp is the token's start (23 of 23 pairs for whisper-base, 18 of
  20 for whisper-tiny, whose timestamps run about 0.2 s late). `experiments/check_spillover_frontend.py`
  holds Whisper's log-mel floor fixed and finds the spill-over changes by at most 0.002 at any layer.
- **CUDA clean-environment run.** `TORCH_INDEX=cu128 scripts/reproduce_clean_env.sh` installs the CUDA
  build in a fresh venv, runs all 50 tests on the GPU and writes `provenance/requirements-lock-cu128.txt`
  (103 pinned distributions).
- **`experiments/tool_examples.py`** regenerates the tool example figures, which had no script.
- Every figure was regenerated from its script. Boxes, counts and numbers are unchanged. Titles
  and row captions follow the current figure text.
- The per-run `provenance.json` files of earlier runs are removed. They no longer described those runs.
- `sound_token` takes one revision per model. README limitations shortened.

## 0.3.1

An independent review of the code and of the experiment scripts, and what it changed.

- **Fresh-clone re-runs work.** A model pass skips a clip only when its saved map exists
  (`experiments/common.py:resume_rows`). Before, the tracked row files made every pass skip all clips
  and the summaries then failed on the missing maps. `run_demand.sh` prepares the set when its cache is
  missing, `demand.py summarise` refuses to write empty results, and a run stopped by the time cap
  exits non-zero.
- **Clean install fixed.** The licence classifier that current setuptools rejects is removed, `scipy`
  is declared in `pyproject.toml`, `ruff` is pinned in `requirements.txt`, and the lock file carries the
  index line it needs. `docs/images/render.py` exits non-zero on a missing font or an overflow.
- **Paired noise-focus differences use the informative clips**, as the per-model ratios already did.
  The whisper-tiny differences change from -0.04 and -0.05 to -0.11 and -0.11, with intervals that
  exclude 0.
- **Black-box word-change map**: a punctuation-only word no longer shifts the alignment.
- **Guards**: Whisper clips over 30 s are refused, `ways latent` needs `--aadd` or `--window`, a band
  count that leaves an empty band is refused, a plug-in with a duplicate name no longer stops the CLI.
- `flag_profile` follows time for any patch length, and `PatchGrid.n_t` uses integer arithmetic. Both
  are unchanged at the default 100 ms.
- **Provenance**: `code_dirty` is recorded, an unpinned model resolves to the snapshot `main` points
  at, `ambiguous_case.py` pins its three models, and its result files hold repository-relative paths.
- Every figure is written through `ways.figure.save_figure` (no dates, fixed SVG ids).
- The sweep animation tries all 376 patches in the order the code erases them, band by band.
- Code comments removed. Field documentation moved into class docstrings. 7 new tests (50 in all).

## 0.3.0

- **cross-check**, a third registered tool: one model, one clip, one transcript, and for every token the
  erasure reading (first-ranked patch, band, share of erasure mass on the degraded audio) next to the
  encoder reading (gradient centre, width, share on the degraded audio at the top layer and the lowest
  share over layers), a verdict per token, the counts by each method and by both, one figure
  (`ways.figure.cross_check_figure`) and a Markdown table (`ways.tools.cross_check.table`). CLI
  `ways cross-check clip.wav --window a b` or `--aadd`.
- The ambiguous-clip experiment runs cross-check on the three models; README section comparing the
  five tokens of the noisy window under both readings.

## 0.2.2

- **The aud-cam counterpart of the controlled figure**: the same clip and the same three models
  (whisper-tiny added to `experiments/ambiguous_case.py`), one lane of per-token gradient saliency per
  model with the erasure test's first patch as a tick, under the controlled figure in the README.
- **experiments/ambiguous_wer.py**: the clean and the degraded transcript of each model on the
  ambiguous clip aligned to the gold, every word placed at its first-ranked erasure patch, the
  shared words joined; explains the WER change (and whisper-base's unchanged WER) through the noisy
  window. Figure and table in the README.

## 0.2.1

One ambiguous clip read in depth, and a per-token view in aud-cam.

- **aud-cam `tokens=True`** (`--tokens`): a gradient saliency per token and encoder layer, one
  backward pass per token from its own surprisal (`ways.latent.token_cam`), a per-token summary
  (`token_cam_summary`: centre, width, share on the degraded stretch, reading against the erasure
  test) and the token figure (`ways.figure.token_cam_figure`).
- The token map is gradient times activation per frame, Grad-CAM without the time pooling. Grad-CAM
  proper (`pooled=True`) is kept for comparison: on these encoders it places every token in the
  middle of the clip, so it is not used for the token view.
- **experiments/ambiguous_case.py**: the controlled noise example (LibriSpeech 2094-142345-0024)
  through whisper-base and moonshine-base, degraded and clean, with erasure, the per-frame map and
  pooled Grad-CAM placed against the spoken order; README section with the figure and the tables.
- Tests: the token summary and figure on synthetic arrays, the tool schema.

## 0.2.0

The two views become a tool framework, and the repository carries no project, machine or user
name anywhere.

- **Tool framework**: `ways.tools` is a package with a contract (`Tool`, `ToolResult`), a registry
  (`register`, `get`, `names`, `schemas`) and one module per tool. The CLI builds one subcommand per
  registered tool, so a plug-in that registers a `Tool` under the `ways.tools` entry-point group
  shows up in `ways tools` and runs as `ways <name>`. `call(name, audio=...)` runs a tool by name
  from a file path and returns a `ToolResult` with `to_json()`.
- **sound-token** and **aud-cam** as registered tools, with `--granular` on aud-cam writing the
  Grad-CAM lane, the spill-over lane and the per-layer summary as their own figures.
- **Figure system**: known damaged spans (one or a list) are drawn dashed and the words are counted
  against them instead of the detector; titles with explicit line breaks wrap per line; a text that
  runs off the page is a layout fault; tick labels outside the axis range are ignored by the check.
- **VoiceBank-DEMAND**: a third dataset with real recorded noise (CC BY 4.0), the noise switched on
  and off inside each sentence from the sample-aligned clean and noisy twins, 100 clips through
  three models under a 30-minute cap each, scored against the clean-recording word times.
- **Provenance**: every model pass of every experiment writes `provenance.json`; paths in public
  files are relative to the repository or `$WAYS_DATA/...`; the experiments read a plain data
  folder (`WAYS_DATA`) whose layout is documented in `experiments/common.py`.
- **Repository hygiene**: CHANGELOG, CONTRIBUTING, SECURITY, `.editorconfig`, a Makefile, ruff
  configuration, and `.gitignore` rules for assistant and editor context files.

## 0.1.0

First release: the STFT-patch erasure operator, three backends (Whisper, Moonshine, any
OpenAI-compatible server), the per-token scores and summaries, AADD degradation context, the
figure system, the latent view, the 100-clip VoxCeleb study, the controlled LibriSpeech set, the
detector-backbone check, the clean-environment execution test and the provenance manifests.
