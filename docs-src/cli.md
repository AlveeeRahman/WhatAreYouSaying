# Command line

Installing the package adds the `ways` command. Every command and option below is read from `ways/cli.py`, and `ways --help` or `ways <command> --help` prints the same list. Audio of any sample rate is read with soundfile, mixed to mono and resampled to 16 kHz.

## ways explain

Per-token erasure attribution for one clip.

```sh
ways explain clip.wav --backend hf-whisper --model openai/whisper-base --aadd --figure --out out/
ways explain clip.wav --backend hf-moonshine --model UsefulSensors/moonshine-base --out out/
ways explain clip.wav --backend openai-compat --base-url http://localhost:8000/v1 --model openai/whisper-large-v3 --out out/
```

| option | default | meaning |
|---|---|---|
| `audio` | | the clip to explain |
| `--backend` | `hf-whisper` | `hf-whisper`, `hf-moonshine` or `openai-compat` |
| `--model` | the backend's default | the model id, for example `openai/whisper-base` |
| `--revision` | latest | the Hugging Face commit of the model, for `hf-*` backends |
| `--base-url` | `http://localhost:8000/v1` | the server, for `openai-compat` |
| `--route` | `transcriptions` | `transcriptions` or `chat`, for `openai-compat` |
| `--patch` | `0.1` | patch length in seconds, a multiple of 0.01 |
| `--bands` | `8` | number of frequency bands |
| `--batch` | `64` | erased patches scored per forward pass |
| `--aadd` | off | also run AADD and report the noise-focus ratio |
| `--aadd-model` | `v3` | the AADD checkpoint |
| `--figure` | off | also draw the clip figure |
| `--out` | `ways_out` | the output folder |
| `-v`, `--verbose` | off | print progress |

It prints the clip summary as JSON, the transcript, and one line per token: its index, the text piece, the centre of its top patch in seconds, the top band, that patch's share of the evidence and the noise-focus ratio. In the output folder it writes:

| file | contents |
|---|---|
| `<clip>.npz` | the attribution arrays, which `Attribution.load` reads back |
| `<clip>.json` | the clip summary and one row per token |
| `<clip>_aadd.npy` | the AADD frames, with `--aadd` |
| `<clip>.png` | the figure, with `--figure` |

## ways latent

Per-layer Grad-CAM and spill-over when the degraded audio is erased. It needs a white-box backend, and either `--aadd` or `--window`, since without a degraded stretch there is nothing to erase.

```sh
ways latent clip.wav --backend hf-whisper --model openai/whisper-base --aadd --figure --out out/
ways latent clip.wav --window 1.2 2.0 --out out/
```

| option | default | meaning |
|---|---|---|
| `audio` | | the clip |
| `--backend` | `hf-whisper` | `hf-whisper` or `hf-moonshine` |
| `--model` | the backend's default | the model id |
| `--revision` | latest | the Hugging Face commit of the model |
| `--aadd` | off | erase the stretches AADD flags |
| `--aadd-model` | `v3` | the AADD checkpoint |
| `--window START_S END_S` | | erase this stretch instead of AADD's flags |
| `--figure` | off | also draw the latent figure |
| `--out` | `ways_out` | the output folder |

It prints one line per encoder layer with the spill-over outside the erased stretch, the shift inside it, the CAM focus and the correlation with loudness. It writes `<clip>_latent.npz`, `<clip>_latent.json`, and `<clip>_latent.png` with `--figure`.

## ways sound-token, ways aud-cam, ways cross-check

Each registered tool is also a command. The three that ship with WAYS:

| command | question it answers | backends |
|---|---|---|
| `ways sound-token` | which audio each word of the transcript used, by erasing the audio patch by patch | any, one or several |
| `ways aud-cam` | what degraded audio does to the model's encoder, layer by layer | white-box |
| `ways cross-check` | the tokens on degraded audio under both readings, the erasure test and the encoder's gradient map | white-box |

```sh
ways sound-token clip.wav --backend hf-whisper=openai/whisper-base --backend hf-moonshine=UsefulSensors/moonshine-base --aadd --out out/clip
ways aud-cam clip.wav --backend hf-whisper=openai/whisper-base --aadd --granular --out out/clip
ways cross-check clip.wav --backend hf-whisper=openai/whisper-base --aadd --reference "the human transcript" --out out/clip
```

| option | meaning |
|---|---|
| `audio` | the clip |
| `--backend KIND=MODEL` | for example `hf-whisper=openai/whisper-base`; repeat it for a tool that takes several backends; default `hf-whisper=openai/whisper-base` |
| `--aadd` | run AADD v3 to find the degraded audio |
| `--window START_S END_S` | a known damaged span instead of AADD's flags |
| `--reference TEXT` | a human transcript, to report the word error rate (`sound-token`, `cross-check`) |
| `--granular` | also write each lane as its own figure (`aud-cam`) |
| `--tokens` | also compute Grad-CAM per token (`aud-cam`) |
| `--title TEXT` | the figure's title |
| `--out PREFIX` | where the result goes, default `ways_out/clip` |

Each prints its result as JSON. [The tool framework](tools.md) explains the tools and how to register a new one, which then appears as a command too.

## ways tools

```sh
ways tools            # each tool's name, whether it needs a white-box backend, and its summary
ways tools --schema   # the function-calling schemas of every tool, as JSON
```

## ways quality

The AADD degradation timeline for one clip, in frames of 20 ms.

```sh
ways quality clip.wav
ways quality clip.wav --aadd-model v3 --out clip_aadd.npy
```

It prints how many frames were flagged at a probability of 0.5 or more, then each flagged stretch with its class, start, end and peak probability. `--out` also saves the frame probabilities as a NumPy file.
