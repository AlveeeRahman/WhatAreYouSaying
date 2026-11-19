"""Black-box backend for any OpenAI-compatible speech server: vLLM (`vllm serve openai/whisper-...`
exposes /v1/audio/transcriptions; audio LLMs such as Qwen2-Audio, Ultravox or Voxtral take audio in
/v1/chat/completions as `input_audio`) and llama.cpp's server (`llama-server` with an mtmd audio
projector accepts the same `input_audio` chat message).

The server only returns text, so the erasure test is scored by re-decoding: erase a patch, send
the erased audio, align the new transcript to the original one word by word, and mark the words
that changed. That is the discrete "did the model's token selection change" form of representation
erasure, and it needs no access to logits.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf

from .base import SR, Backend, Transcript


def _wav_bytes(wav: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, np.float32), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class OpenAICompatBackend(Backend):
    name = "openai-compat"
    supports_surprisal = False

    def __init__(self, base_url: str = "http://localhost:8000/v1", model_id: str = "openai/whisper-base",
                 api_key: str = "EMPTY", route: str = "transcriptions", language: str = "en",
                 prompt: str = "Transcribe the audio exactly, as plain text.", timeout: float = 120.0):
        """route = "transcriptions" (POST /audio/transcriptions, Whisper-style ASR servers) or
        "chat" (POST /chat/completions with an input_audio part, audio LLMs on vLLM / llama.cpp)."""
        from openai import OpenAI

        assert route in ("transcriptions", "chat"), route
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model_id, self.route, self.language, self.prompt = model_id, route, language, prompt
        self.base_url = base_url

    def _text(self, wav: np.ndarray) -> str:
        data = _wav_bytes(wav)
        if self.route == "transcriptions":
            r = self.client.audio.transcriptions.create(model=self.model_id, file=("clip.wav", data, "audio/wav"),
                                                        language=self.language, response_format="json", temperature=0)
            return (r.text or "").strip()
        msg = [{"role": "user", "content": [
            {"type": "text", "text": self.prompt},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": "wav"}}]}]
        r = self.client.chat.completions.create(model=self.model_id, messages=msg, temperature=0)
        return (r.choices[0].message.content or "").strip()

    def transcribe(self, wav: np.ndarray) -> Transcript:
        text = self._text(wav)
        words = text.split()
        pieces = [" " + w for w in words]
        n = len(words)
        ids = np.arange(n + 2, dtype=np.int64)
        tr = Transcript(ids=ids, keep=np.arange(n, dtype=np.int64), pieces=pieces, text=text, times=None)
        tr.group_words()
        return tr

    def describe(self) -> dict:
        d = super().describe()
        d.update(base_url=self.base_url, route=self.route, scoring="selection (re-decode and align)")
        return d
