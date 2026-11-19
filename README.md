<p align="center">
  <img src="docs/images/ways_icon.png" width="180" alt="WhatAreYouSaying: a large ear with a hand cupped behind it and three sound arcs coming in">
</p>
<h1 align="center">WhatAreYouSaying</h1>
<p align="center"><sub>Which audio did the speech model use for each word it wrote, and was that audio clean?</sub></p>

<p align="center">
  <img src="figures/method_whisper_base.png" width="860" alt="Representation erasure on audio for whisper-base on one clip: the erasure grid over the spectrogram with one patch highlighted, the patch before and after erasure in the model's log-mel input, the surprisal change of every token for that patch, and the full map of the most affected token with the AADD-flagged frames marked above it">
  <br><sub>The method on one clip (whisper-base). Erase one patch of the spectrogram, re-score the model's own transcript, note which token got less likely. Repeat for every patch.</sub>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/tests-14%20passed-brightgreen.svg" alt="tests: 14 passed">
  <img src="https://img.shields.io/badge/CI-not%20set%20up%20yet-lightgrey.svg" alt="CI: not set up yet">
</p>

WhatAreYouSaying (WAYS) tells you, for every token a speech-to-text model wrote, which 100 ms of
audio in which frequency band it leaned on, and whether that audio was clean or degraded. It works
with local Whisper and Moonshine weights, and with any OpenAI-compatible speech server such as vLLM
or llama.cpp. No training, no labels, one forward pass per erased patch.

It builds on two works and nothing else: representation erasure by Li, Monroe and Jurafsky
[[1]](#ref-1), which measures what a neural network uses by deleting parts of its input and
watching the likelihood of the right answer, and AADD by Arina Gertel [[2]](#ref-2), which says
what is wrong with a recording and where, every 20 ms. The speech models are Whisper
[[3]](#ref-3) and Moonshine [[4]](#ref-4). The reproducibility steps follow Dr Tanu Malik's work
on re-executable research [[5]](#ref-5) [[6]](#ref-6).

MIT licence. This is the framework behind the earlier per-token attribution study in
the pipeline repository,
rewritten so that it does not depend on that pipeline and so that the same erasure operator serves
every model.

**Contents:** [What it answers](#what-it-answers) · [How it works](#how-it-works) · [What I verified](#what-i-verified) · [Results on 100 random test clips](#results-on-100-random-test-clips) · [Does the detector matter](#does-the-detector-matter) · [Controlled degradation](#controlled-degradation-on-librispeech) · [Install and use](#install-and-use) · [Backends](#backends-local-weights-vllm-llamacpp) · [Reproducibility](#reproducibility) · [Limitations and future work](#limitations-and-future-work) · [Status](#status) · [References](#references) · [Repository layout](#repository-layout) · [Credits and licence](#credits-and-licence)

<details><summary>Stats for nerds</summary>

| | |
|---|---|
| erasure grid | 100 ms x 8 mel-spaced bands (0, 372, 745, 1128, 1657, 2433, 3571, 5244, 8000 Hz), on Whisper's STFT (25 ms Hann window, 10 ms hop) |
| erased patch, measured in Whisper's log-mel input | down 33 to 36 dB on average over three real clips, worst patch 17 dB, which leaves it below each recording's own 10th-percentile background level |
| scores per token and patch | surprisal change in nats, Li et al.'s relative change, argmax flip; black-box servers get a 0/1 word-change map |
| cost | one teacher-forced forward pass per patch, batched 64 at a time; 4.3 s per clip for moonshine-base, about 15 s for whisper-base on an RTX 4000 Ada (20 GB) |
| sample | 100 clips, simple random sample with seed 8001 from the saved 1 500-clip test slice (sha256 `dfb0451d…`), total 862 s of audio |
| detectors | AADD v3.1 (Whisper-small backbone, 88.6 M parameters, 20 ms), v3-lite (CRNN, 1.3 M, 20 ms), v2 (EfficientNet-B0, 4.0 M, 1 s) |
| tests | 12 CPU checks of the operator and the formulas, 2 end-to-end HTTP checks against a mock OpenAI-compatible server |

</details>

## What it answers

A speech model turns a waveform into tokens. When it gets a word wrong, or right against the odds,
I want to know which part of the sound it was listening to, and whether that part was speech or
noise. Attention maps and gradients answer this only loosely: in the earlier study, gradient x
input on whisper-tiny kept 92 % of its ranking after the decoder was re-initialised, so it was
mostly reflecting input energy, not the model. Erasure answers it directly: remove the audio, see
what changes.

WAYS gives three things per clip:

* a map per token over time and frequency, saying how much each patch of audio supported it;
* the top patch per token, drawn and labelled on the spectrogram, with the transcript printed below it in the same colours and numbers (figure below);
* a noise-focus ratio per token and per clip: how much of the token's evidence sits in audio that
  AADD flags as degraded, relative to how much of the clip is flagged. 1 means no preference, 2 means
  the token leaned on degraded audio twice as much as chance.

<p align="center">
  <img src="figures/three_models__id10288_FIxC_PNJNkc_00003.png" width="860" alt="The same clip through whisper-tiny, whisper-base and moonshine-base: for each model the spectrogram with each token's most important patch boxed in the token's colour, and the tokens placed on the audio segment they used most">
  <br><sub>One clip, three models. The spectrogram and the AADD strip are identical in every row; only the boxes and the transcript strip change. Same colour and number = same token.</sub>
</p>

## How it works

### Erasing a patch

Li et al. [[1]](#ref-1) erase one dimension of the input by setting it to zero, then measure the
relative change in the negative log-likelihood of the correct label. For audio the "dimension" is a
patch of the time-frequency plane. Let X = STFT(x) with Whisper's analysis settings (window 400
samples, hop 160, Hann, 16 kHz). Patch P = (b, t) covers STFT frames [10t, 10t + 10) and the
frequency bins of band b. Erasing it is

$$
\tilde{X}_P = X \odot (1 - \mathbb{1}_P), \qquad x \backslash P = \mathrm{ISTFT}(\tilde{X}_P)
$$

The erased waveform goes to the model through its own front end, so one operator serves Whisper
(log-mel input), Moonshine (raw waveform input) and a remote server (a WAV file over HTTP).

Zeroing coefficients and resynthesising is not exact: the 25 ms windows overlap by 75 %, so the
new signal's own STFT regains some energy in the patch from the frames and bins around it. WAYS
applies the projection three times (zero, resynthesise, re-analyse). Measured in Whisper's log-mel
on three real clips, the patch falls by 33 to 36 dB on average and by at least 17 dB in the worst
case, which puts it below the clip's own 10th-percentile background. Without resynthesis the drop
would be 56 dB; the residual is the price of staying in the waveform domain. The waveform outside
the patch's time span is unchanged to 1e-5, and only the two patches that share a window edge with
P lose a little of their own energy (up to about 10 % of P's effect).

### Scoring a token

Let y = (y₁ … y_K) be the model's own greedy transcript and S_k(x) = −log p(y_k | y_<k, x) the
surprisal of token k with the earlier tokens held fixed (teacher forcing). For every patch P:

$$
\Delta_k(P) = S_k(x \backslash P) - S_k(x), \qquad
r_k(P) = \frac{\Delta_k(P)}{S_k(x)}, \qquad
\mathrm{flip}_k(P) = \big[\arg\max_v p(v \mid y_{<k}, x \backslash P) \neq y_k\big]
$$

Δ is in nats and is positive when the patch supported the token. r is Li et al.'s equation 1 (the
relative change of the negative log-likelihood), with the sign chosen so that a supporting patch is
positive. flip is their discrete question, "does the decision change", asked per token. All three
come out of the same forward pass. Dividing by S_k(x) is a per-token constant, so every within-token
summary below is the same whether it is computed from Δ or from r; r only matters when tokens are
compared with each other, and it is unstable for tokens the model was nearly certain of (S_k ≈ 0),
so the summaries use Δ.

A server that returns only text cannot give S_k. There WAYS re-decodes x\P, aligns the new
transcript to y word by word with jiwer, and marks each original word that is not inside an
"equal" alignment chunk. That is the "relative change in evaluation metric" form of erasure from
the same paper, at word level, 0 or 1 per patch.

### From a map to a sentence

For token k, the non-negative mass is m_k(P) = max(Δ_k(P), 0) / Σ_P max(Δ_k(P), 0). From it:

* time profile Σ_b m_k(b, t) and band shares Σ_t m_k(b, t);
* top patch = argmax m_k, top share = the largest time-profile value;
* for Whisper, the distance between the top patch's centre and the token's own timestamp;
* noise-focus ratio. With f(t) the fraction of 20 ms frames in patch t that AADD flags (any class
  at probability ≥ 0.5) and f̄ its mean over the clip,

$$
\mathrm{NFR}_k = \frac{\sum_t \big(\sum_b m_k(b, t)\big)\, f(t)}{\bar{f}}
$$

NFR is 1 when a token spreads its evidence without regard to the flags, above 1 when it leans on
flagged audio, below 1 when it avoids it. It is undefined when the clip has no flagged frames and
trivially 1 when every frame is flagged, so the tables report it on the clips in between
("informative" clips) as well as on all clips. Uncertainty is a 10 000-resample bootstrap over
clips (seed 8001); model differences are paired on the same clips.

## What I verified

Every formula above has a test in `tests/`, and the numbers below are what those tests and the
smoke runs measured on this machine. `pytest tests -q` runs the first twelve in about one second
on a CPU.

| check | how | result |
|---|---|---|
| ISTFT inverts STFT | random signal through both | max error 1e-5 |
| bands are disjoint and cover 0 to 8 kHz | bin ranges | 9 edges, 201 bins, no gaps |
| erasing a 550 Hz tone from band 1 | energy in the patch before and after | 19 dB down; the band-0 neighbour and the waveform outside the patch unchanged |
| erasing white noise in every band | same | 12.5 to 19.6 dB down |
| the erased patch in the model's input | Whisper log-mel, 200 random patches on each of 3 real clips | 33 to 36 dB down on average, 17 dB worst, below the clip's own background |
| the synthetic backend recovers the planted patch | a fake model whose surprisal rises by exactly 2 nats when its patch is erased | top patch found for both tokens; Δ within 0.1 of 2.0; r = Δ / S₀ exact; exactly one flip; other patches move less than 0.3 |
| mass normalisation and band shares sum to 1 | `token_mass`, `token_summary` | exact |
| NFR = 1 for uniform mass, 2 when all mass is on the flagged half | synthetic flags | exact |
| WER counts and normalisation | known sentences | S, D, I, N as expected; punctuation and case ignored |
| teacher-forced surprisal equals the generation scores | whisper-base and moonshine-base on the example clip | max difference 6.5e-6 (Moonshine); Whisper identical by construction |
| scoring twice gives the same number | same clip twice | difference 0.0 |
| black-box word-change map agrees with the white-box flips | mock OpenAI-compatible server backed by whisper-tiny, both routes | every changed word sits at or after a flipped token, 2 tests pass |
| AADD through the Python API equals the CLI scan | first sample clip, v3.1 | max difference 5e-5, flag agreement 100 % |

The last line matters for the pipeline results: the AADD numbers used there came from the command
line tool, and the ones here from `aadd.infer.Detector` on the same waveform.

## Results on 100 random test clips

Three small models, the same 100 clips: a simple random sample (seed 8001) from the saved
1 500-clip test slice of the pipeline, 862 s of VoxCeleb interview audio with the SLUE human
transcripts as reference. Every patch of every clip was erased once per model: 695 patches per clip
on average, 69 456 teacher-forced forward passes per model, 7 s per clip for
whisper-tiny, 14 s for whisper-base and 4 s for moonshine-base on the GPU in
[Reproducibility](#reproducibility). Brackets are 95 % bootstrap intervals over clips.

| model | tokens | WER (mean over clips) | corpus WER | top-patch share | tokens with any flip | noise-focus ratio (v3) | focus, informative clips |
|---|---|---|---|---|---|---|---|
| whisper-tiny | 4034 | 0.35 [0.22, 0.55] | 0.383 | 0.30 [0.29, 0.32] | 0.45 [0.41, 0.48] | 0.97 [0.92, 1.03] | 0.95 [0.87, 1.04] (n=46) |
| whisper-base | 3267 | 0.20 [0.17, 0.22] | 0.182 | 0.26 [0.25, 0.27] | 0.33 [0.30, 0.36] | 1.01 [0.95, 1.10] | 1.02 [0.91, 1.17] (n=46) |
| moonshine-base | 4055 | 0.32 [0.16, 0.57] | 0.256 | 0.31 [0.29, 0.33] | 0.29 [0.26, 0.32] | 1.02 [0.96, 1.09] | 1.04 [0.93, 1.15] (n=46) |

| paired difference | WER | noise-focus ratio | top-patch share |
|---|---|---|---|
| whisper-base - moonshine-base | -0.124 [-0.380, +0.035] | -0.008 [-0.097, +0.077] | -0.050 [-0.069, -0.032] |
| whisper-tiny - whisper-base | +0.151 [+0.023, +0.355] | -0.042 [-0.089, -0.004] | +0.043 [+0.029, +0.058] |
| whisper-tiny - moonshine-base | +0.027 [-0.266, +0.305] | -0.050 [-0.131, +0.016] | -0.007 [-0.031, +0.017] |

<p align="center">
  <img src="figures/comparison_100clips.png" width="860" alt="Three panels: word error rate per clip for the three models; the noise-focus ratio per clip on the informative clips, centred on 1; the mean share of each token's evidence in each of the eight frequency bands">
</p>

What this says, in order of how sure I am:

* **The maps are concentrated and model-specific.** A token's single most important 100 ms x band
  patch carries 30% (whisper-tiny), 26% (whisper-base) and 31% (moonshine-base) of its
  mass on average, out of about 700 patches. The two Whisper sizes differ on this by a paired
  +0.043 [+0.029, +0.058]: the smaller model leans harder on single patches.
* **Whisper's maps land on the token's own time.** The distance between a token's top patch and its
  cross-attention timestamp has a median of 0.12 s for whisper-base (80% of tokens within 0.3 s)
  and 0.31 s for whisper-tiny (71%). Moonshine has no timestamps to compare against.
* **Evidence sits where the voice is.** All three models put a quarter of each token's mass in the
  0.4 to 0.7 kHz band and almost nothing above 3.6 kHz (table below). Moonshine uses the lowest band
  (0 to 0.4 kHz) more than Whisper does, 0.19 against 0.14.
* **Single-patch erasure flips a token's choice in a minority of cases**: 45% of whisper-tiny
  tokens have at least one patch whose erasure changes the argmax, 33% for whisper-base and
  29% for moonshine-base. Most tokens survive the loss of any one patch.
* **On this data no model prefers degraded audio.** The noise-focus ratio is 0.95, 1.02 and
  1.04 on the 46 clips where it is informative, with intervals that all contain 1, and the
  paired differences between models contain 0. The reason is the data as much as the models: AADD
  flags 59% of all frames, every frame in 32 of the 100 clips and none in 22, so the
  flag carries little within-clip information on VoxCeleb. That is why the controlled experiment
  below exists.
* **WER.** whisper-base 0.20 per clip (0.182 corpus), moonshine-base 0.32 (0.256),
  whisper-tiny 0.35 (0.383). The per-clip means of moonshine and whisper-tiny have wide intervals
  because each has one clip where the model looped and produced far more words than the reference.

| model | 0.0-0.4 kHz | 0.4-0.7 kHz | 0.7-1.1 kHz | 1.1-1.7 kHz | 1.7-2.4 kHz | 2.4-3.6 kHz | 3.6-5.2 kHz | 5.2-8.0 kHz |
|---|---|---|---|---|---|---|---|---|
| whisper-tiny | 0.15 | 0.26 | 0.08 | 0.11 | 0.14 | 0.12 | 0.06 | 0.08 |
| whisper-base | 0.14 | 0.25 | 0.08 | 0.13 | 0.14 | 0.11 | 0.07 | 0.09 |
| moonshine-base | 0.19 | 0.26 | 0.06 | 0.11 | 0.15 | 0.10 | 0.05 | 0.09 |

<p align="center">
  <img src="figures/vox_clean__id10271_gO805KoL2RM_00006.png" width="860" alt="A VoxCeleb clip that AADD rates clean, through whisper-tiny, whisper-base and moonshine-base: spectrogram with each token's most important patch boxed and labelled, and the transcript strip">
  <br><sub>A clean clip (AADD flags no frame). whisper-base and moonshine-base agree on 25 of 26 words; whisper-tiny drops "kind of" and "it".</sub>
</p>
<p align="center">
  <img src="figures/vox_degraded__id10295_K1gPy1UKmBE_00007.png" width="860" alt="A VoxCeleb clip that AADD rates degraded in every frame, through the same three models">
  <br><sub>A degraded clip (AADD flags every frame as noise). All three models get it with one error; the boxes still sit on the voiced harmonics, not on the noise floor.</sub>
</p>

## Does the detector matter

AADD v3.1 has a Whisper-small encoder inside it. If the noise-focus result depended on that, a
Whisper model being attributed and the detector judging its audio would share a backbone. So the
same question was asked with three detectors that share nothing but the task: v3.1 (Whisper-small,
88.6 M parameters, 20 ms frames, 12 classes), v3-lite (a 1.3 M-parameter CRNN distilled from v3.1,
same grid) and v2 (EfficientNet-B0 on log-mel, 4.0 M parameters, 1 s segments, 7 synthetic classes).
Each was run on the same 100 waveforms through the AADD Python API; the v3.1 run reproduces the
pipeline's CLI scan to 5e-5. The attribution maps are the same; only the flag changes.

| detector | whisper-tiny | whisper-base | moonshine-base | paired base - moonshine |
|---|---|---|---|---|
| v3 | 0.95 [0.87, 1.04] (n=46) | 1.02 [0.91, 1.17] (n=46) | 1.04 [0.93, 1.15] (n=46) | -0.013 [-0.163, +0.127] |
| v3-lite | 1.00 [0.94, 1.06] (n=41) | 0.99 [0.94, 1.05] (n=41) | 1.00 [0.95, 1.06] (n=41) | -0.012 [-0.046, +0.019] |
| v2 | 0.92 [0.84, 1.01] (n=51) | 0.92 [0.83, 1.01] (n=51) | 0.91 [0.82, 1.00] (n=51) | +0.005 [-0.059, +0.067] |

The answer does not move: every interval contains 1 and every paired difference contains 0. The
number of informative clips changes with the detector (v2 flags fewer frames, so more clips have a
mix), but the conclusion is the same under all three.

## Controlled degradation on LibriSpeech

<!-- CONTROLLED -->

## Install and use

```bash
git clone https://github.com/AlveeeRahman/WhatAreYouSaying && cd WhatAreYouSaying
python -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt   # or /cpu
pip install --no-deps -e .
pytest tests -q
```

One clip, local weights, with the AADD context and the figure:

```bash
ways explain clip.wav --backend hf-whisper --model openai/whisper-base --aadd --figure --out out/
ways explain clip.wav --backend hf-moonshine --model UsefulSensors/moonshine-base --aadd --figure --out out/
ways quality clip.wav                      # just the AADD timeline
```

From Python:

```python
import soundfile as sf
from ways import explain, load_backend
from ways.metrics import token_summary, clip_summary
from ways.quality import Quality

wav, sr = sf.read("clip.wav", dtype="float32")          # 16 kHz mono
att = explain(load_backend("hf-whisper", "openai/whisper-base"), wav)   # delta, rel, flip: [tokens, 8 bands, patches]
probs = Quality("v3").frames(wav)                        # AADD, [frames of 20 ms, 12 classes]
rows = token_summary(att, probs)                         # one dict per token: top patch, band shares, focus ratio
print(clip_summary(att, rows, probs))
```

`att.save(path)` / `Attribution.load(path)` keep the arrays as a compressed npz with the
transcript and grid in a JSON field.

## Backends: local weights, vLLM, llama.cpp

| backend | reached how | score | what it was tested against |
|---|---|---|---|
| `hf-whisper` | transformers, in process | Δ, r, flip | whisper-tiny, whisper-base, 100 clips each |
| `hf-moonshine` | transformers, in process | Δ, r, flip | moonshine-base, 100 clips |
| `openai-compat` | HTTP, `/v1/audio/transcriptions` or `/v1/chat/completions` with `input_audio` | word changed, 0/1 | `tests/mock_server.py` (a local server that answers both routes with whisper-tiny) |

```bash
vllm serve openai/whisper-large-v3 --port 8000
ways explain clip.wav --backend openai-compat --route transcriptions --base-url http://localhost:8000/v1 \
     --model openai/whisper-large-v3 --patch 0.2 --bands 4 --out out/

llama-server -hf ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF --port 8080
ways explain clip.wav --backend openai-compat --route chat --base-url http://localhost:8080/v1 \
     --model ultravox --patch 0.2 --bands 4 --out out/
```

The black-box path costs one request per patch, so a coarser grid is the sensible default there.
I have not run a live vLLM or llama.cpp server for this repository; the client is the official
`openai` package and the request shapes are the ones both servers document. Details and the
chat-route prompt are in [docs/backends.md](docs/backends.md).

## Reproducibility

I followed the steps Dr Malik's group teaches for re-executable research [[5]](#ref-5) and the
three-layer dependency view of her 2026 paper with Vangala, Adibifar and Gehani [[6]](#ref-6),
then tested on real hardware. The full checklist is [docs/reproducibility.md](docs/reproducibility.md).

* **Fixed, hashed inputs.** The 100 clips are `default_rng(8001).choice(1500, 100, replace=False)` on the
  saved test slice (sha256 `dfb0451d8ea3…`), which is disjoint from the development and load slices;
  the script asserts that and never reads load-pool audio. Clip ids: `experiments/sample100.csv`.
* **Pinned claimed dependencies.** `requirements.txt` pins every version. AADD is not on PyPI; it is
  pinned to git commit `c6a2debc`. My first draft said `aadd==3.1.0`, and the clean-environment run
  below failed on it. That is exactly the claimed-versus-working gap the paper measures, caught
  by the test the paper recommends.
* **Runtime provenance written by the code.** Each run writes `provenance.json` with the git commit,
  Python, torch and CUDA versions, the GPU, input hashes, model snapshot commits, seeds, the claimed
  dependency list and the runtime dependency list (every distribution behind an imported module),
  plus the difference between the two.
* **Clean-environment execution test.** `scripts/reproduce_clean_env.sh` builds a fresh venv with no
  system packages, installs only `requirements.txt` (CPU torch), installs the package, runs the CPU
  tests, and records what loaded. Result on this machine: passed; 14 claimed packages, 16
  distributions imported by the tests, 11 of them transitive (`provenance/clean_env_run.json`).
* **Real hardware.** Everything in `experiments/results/` ran on one NVIDIA RTX 4000 Ada (20 GB),
  Ubuntu, Python 3.12.3, torch 2.11.0+cu128, transformers 5.5.4. Continuous integration is not set
  up yet; the command a workflow would run is `pytest tests -q -m "not gpu and not slow"`.

## Limitations and future work

* **The erasure is not a hard zero.** 33 to 36 dB down on average, 17 dB worst case, because of
  the window overlap. For a model that reads the waveform there is no cleaner operator at this
  resolution; for Whisper alone one could set the log-mel patch to the floor, but then the two
  model families would not get the same test.
* **Teacher forcing scores the model's own transcript.** Δ says how much a patch supported what the
  model wrote, not what it should have written. A wrong token with a confident map is still a wrong
  token; the WER column is kept next to the maps for that reason.
* **AADD is a detector, not ground truth.** It was trained on synthetic degradations of read speech;
  on these real interview clips it flags 59 % of frames and every frame in a third of the clips.
  The noise-focus ratio is reported on the clips where it is informative, and under three detectors.
* **Moonshine has no timestamps**, so the distance-to-timestamp summary exists only for Whisper.
* **Neighbouring patches share an edge.** A patch's two time neighbours lose up to about 10 % of
  its effect; maps are smooth across those edges by construction.
* **No live vLLM or llama.cpp run yet**, only the mock server. Running one is the next test.
* **100 clips, three small models.** Enough for paired comparisons with bootstrap intervals, not
  for claims about large models or other languages.

## Status

* Done: operator, three backends, scores, summaries, AADD context, CLI, figures, tests, 100-clip
  runs for three models, three-detector check, clean-environment run, provenance.
* Next: CI workflow once Actions minutes are available (the test command is fixed); a live vLLM
  transcription run; the same 100 clips on Modal if a larger model is wanted.

## References

<a id="ref-1"></a>[1] J. Li, W. Monroe, and D. Jurafsky, "Understanding neural networks through representation erasure," *arXiv:1612.08220*, Dec. 2016. [arXiv](https://arxiv.org/abs/1612.08220)

<a id="ref-2"></a>[2] A. Gertel, "AADD: Automatic Audio Degradation Detection," software, version 3.1.0, 2026, MIT licence. [GitHub](https://github.com/gertelrina/aadd) · [doi:10.5281/zenodo.24352878](https://doi.org/10.5281/zenodo.24352878)

<a id="ref-3"></a>[3] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *Proc. ICML*, 2023. [arXiv](https://arxiv.org/abs/2212.04356)

<a id="ref-4"></a>[4] N. Jeffries, E. King, M. Kudlur, G. Nicholson, J. Wang, and P. Warden, "Moonshine: Speech recognition for live transcription and voice commands," *arXiv:2410.15608*, 2024. [arXiv](https://arxiv.org/abs/2410.15608)

<a id="ref-5"></a>[5] D. H. Ton That, G. Fils, Z. Yuan, and T. Malik, "Sciunits: Reusable research objects," in *Proc. IEEE 13th Int. Conf. on e-Science*, 2017. [arXiv](https://arxiv.org/abs/1707.05731)

<a id="ref-6"></a>[6] B. P. Vangala, A. Adibifar, A. Gehani, and T. Malik, "AI-generated code is not reproducible (yet): An empirical study of dependency gaps in LLM-based coding agents," *arXiv:2512.22387*, v3, Mar. 2026. [arXiv](https://arxiv.org/abs/2512.22387)

## Repository layout

| Path | What it holds |
|---|---|
| `ways/erasure.py` | the STFT patch grid and the erasure operator |
| `ways/attribution.py` | `explain()`: Δ, r and flip per token and patch; the black-box word-change map |
| `ways/backends/` | `hf_whisper.py`, `hf_moonshine.py`, `openai_compat.py` (vLLM, llama.cpp) |
| `ways/quality.py` | AADD wrapper, flagged mask, degraded intervals |
| `ways/metrics.py` | token and clip summaries, noise-focus ratio, WER, bootstrap |
| `ways/plots.py` | the figures, in AADD's inspector style |
| `ways/provenance.py` | the provenance manifest (claimed and runtime dependencies, hashes, model commits) |
| `ways/cli.py` | `ways explain`, `ways quality` |
| `tests/` | operator and formula checks; mock OpenAI-compatible server and the HTTP test |
| `experiments/` | sample selection, the three-model runs, the three-detector run, comparison tables, the example clip |
| `experiments/results/` | per-clip rows, per-token tables, `comparison.json`, `comparison.md`, provenance (npz maps are not committed) |
| `figures/` | the method figure, per-model clip figures, the three-model figure, the comparison figure |
| `docs/` | backends, reproducibility checklist, the icon and its drawing script |
| `provenance/` | clean-environment run log and runtime dependency set |
| `scripts/reproduce_clean_env.sh` | the clean-environment execution test |

## Credits and licence

WhatAreYouSaying is released under the MIT licence (see `LICENSE`). AADD [[2]](#ref-2) is MIT and
keeps its licence; Whisper weights are Apache-2.0, Moonshine weights MIT, transformers Apache-2.0.
The VoxCeleb audio and the SLUE transcripts used in `experiments/` are not part of this repository;
only clip ids, per-clip and per-token numbers, and figures are. The attribution design started in
an earlier study of a speech pipeline, whose saved test slice
and AADD scan this work samples from.
