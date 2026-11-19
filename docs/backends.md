# Backends: local weights, vLLM, llama.cpp

WAYS asks a speech-to-text model two things: transcribe this audio, and (if it can) tell me how
surprised you are by each token of a fixed transcript. The second is only possible with access to
the logits. So there are two kinds of backend and two kinds of score.

| backend | how it is reached | score per patch | needs |
|---|---|---|---|
| `hf-whisper` | Hugging Face weights in process | surprisal change `delta_k(P)`, plus the argmax flip | GPU recommended |
| `hf-moonshine` | Hugging Face weights in process | same | GPU recommended |
| `openai-compat` | HTTP, any OpenAI-compatible server | word changed under re-decoding, 0 or 1 | a running server |

The erasure operator is the same in all three: it acts on the waveform (through the STFT), so the
erased audio can be fed to a local front end or sent over the wire as a WAV file.

## Local Hugging Face weights

```bash
ways explain clip.wav --backend hf-whisper   --model openai/whisper-base         --aadd --figure --out out/
ways explain clip.wav --backend hf-moonshine --model UsefulSensors/moonshine-base --aadd --figure --out out/
```

Any Whisper checkpoint on the Hub works (`openai/whisper-tiny` to `openai/whisper-large-v3`, and
fine-tunes such as `AventIQ-AI/whisper-audio-to-text`). Whisper timestamps come from the
cross-attention alignment heads when the checkpoint ships them; Moonshine has no timestamps, so the
"distance to the token's own time" summary is only reported for Whisper.

## vLLM

vLLM serves Whisper through the OpenAI transcription route:

```bash
vllm serve openai/whisper-large-v3 --port 8000
ways explain clip.wav --backend openai-compat --route transcriptions \
     --base-url http://localhost:8000/v1 --model openai/whisper-large-v3 --patch 0.2 --bands 4 --out out/
```

Audio LLMs on vLLM (Qwen2-Audio, Ultravox, Voxtral and others) take the audio inside a chat
message. Use `--route chat`; the audio goes as a base64 `input_audio` part with the prompt
"Transcribe the audio exactly, as plain text."

```bash
vllm serve Qwen/Qwen2-Audio-7B-Instruct --port 8000
ways explain clip.wav --backend openai-compat --route chat --base-url http://localhost:8000/v1 \
     --model Qwen/Qwen2-Audio-7B-Instruct --patch 0.2 --bands 4 --out out/
```

## llama.cpp

`llama-server` with a multimodal (mtmd) audio projector accepts the same `input_audio` chat part, so
the chat route above works unchanged:

```bash
llama-server -hf ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF --port 8080
ways explain clip.wav --backend openai-compat --route chat --base-url http://localhost:8080/v1 \
     --model ultravox --patch 0.2 --bands 4 --out out/
```

## What the black-box score means

With no logits, the test is the discrete one from Li et al.: does the model's output change when
the patch is gone? WAYS re-decodes the erased audio, aligns the new transcript to the original with
`jiwer`, and marks every original word that is not inside an "equal" alignment chunk. The result is
a 0/1 map per word and patch. It is coarser than the surprisal map and costs one request per patch,
so pick a coarser grid (`--patch 0.2 --bands 4` gives a quarter of the requests of the default).

## What has been tested

The HTTP path was exercised end to end against `tests/mock_server.py`, a small server that answers
both routes with a local whisper-tiny (`pytest tests/test_openai_compat.py`). The test checks that
every word the black-box score marks as changed sits at or after a token whose argmax flipped in the
white-box score of the same model on the same erased audio. A live vLLM or llama.cpp server has not
been run here; the request and response shapes follow the OpenAI API that both servers implement,
and the client is the official `openai` package.
