# Changelog

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
