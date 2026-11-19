"""Backend protocol. A backend wraps one speech-to-text model.

Every backend can transcribe. A white-box backend (local Hugging Face weights) can also score a
fixed token sequence under teacher forcing, which gives the per-token surprisal that the erasure
score needs. A black-box backend (an OpenAI-compatible HTTP server such as vLLM or llama.cpp)
only transcribes, so it is scored by re-decoding the erased audio and counting which words change
(the "token selection" variant of the same erasure test).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 16000


@dataclass
class Transcript:
    """One decoded hypothesis.

    ids    : full decoder sequence including special tokens (what the model actually generated)
    keep   : positions p into ids[1:] whose target ids[p+1] is a text token; the k-th scored token
             is ids[keep[k] + 1]
    pieces : display string of each kept token; a leading space marks the start of a word
    times  : optional seconds per kept token (Whisper cross-attention timestamps), else None
    text   : the detokenised hypothesis
    """

    ids: np.ndarray
    keep: np.ndarray
    pieces: list[str]
    text: str
    times: np.ndarray | None = None
    words: list[dict] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.keep)

    def token_ids(self) -> np.ndarray:
        return self.ids[self.keep + 1]

    def group_words(self) -> list[dict]:
        """Merge pieces into words: dicts with word, toks (indices into keep), t0, t1 if timed."""
        words: list[dict] = []
        for k, s in enumerate(self.pieces):
            new = (not words) or s.startswith(" ") or s.startswith("▁")
            if new:
                words.append(dict(word=s, toks=[k]))
            else:
                words[-1]["word"] += s
                words[-1]["toks"].append(k)
        for w in words:
            w["word"] = w["word"].replace("▁", " ").strip()
            if self.times is not None:
                w["t0"] = float(self.times[w["toks"][0]])
                w["t1"] = float(self.times[w["toks"][-1]])
        self.words = words
        return words


class Backend:
    name: str = "base"
    model_id: str = ""
    revision: str | None = None
    supports_surprisal: bool = False

    def transcribe(self, wav: np.ndarray) -> Transcript:
        raise NotImplementedError

    def transcribe_batch(self, wavs: np.ndarray) -> list[Transcript]:
        return [self.transcribe(w) for w in wavs]

    def surprisal(self, wavs: np.ndarray, tr: Transcript) -> tuple[np.ndarray, np.ndarray]:
        """wavs [B, N] -> (S [B, K], argmax [B, K]).

        S[b, k] = -log p(y_k | y_<k, wavs[b]) with the transcript's own previous tokens held fixed
        (teacher forcing); argmax[b, k] is the token the model would pick at step k given wavs[b].
        """
        raise NotImplementedError(f"{self.name} is a black-box backend; use selection scoring")

    def describe(self) -> dict:
        return dict(backend=self.name, model=self.model_id, revision=self.revision,
                    supports_surprisal=self.supports_surprisal)
