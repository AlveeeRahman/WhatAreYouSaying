"""Whisper (Hugging Face transformers) white-box backend."""
from __future__ import annotations

import numpy as np
import torch

from .base import SR, Backend, Transcript


class WhisperBackend(Backend):
    name = "hf-whisper"
    supports_surprisal = True

    def __init__(self, model_id: str = "openai/whisper-base", revision: str | None = None,
                 device: str | None = None, language: str = "en"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id, self.revision, self.language = model_id, revision, language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = WhisperProcessor.from_pretrained(model_id, revision=revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=torch.float32, attn_implementation="eager").to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.eot = self.proc.tokenizer.eos_token_id
        self.has_alignment_heads = getattr(self.model.generation_config, "alignment_heads", None) is not None

    def features(self, wavs: np.ndarray):
        f = self.proc.feature_extractor(list(np.asarray(wavs, np.float32)), sampling_rate=SR, return_tensors="pt",
                                        return_attention_mask=True, device=self.device)
        return f.input_features.to(self.device), f.attention_mask.to(self.device)

    def transcribe(self, wav: np.ndarray) -> Transcript:
        feats, am = self.features(np.asarray(wav, np.float32)[None])
        with torch.no_grad():
            out = self.model.generate(feats, attention_mask=am, language=self.language, task="transcribe",
                                      return_token_timestamps=self.has_alignment_heads, return_dict_in_generate=True)
        seq = out["sequences"][0].cpu().numpy()
        times = out["token_timestamps"][0].float().cpu().numpy() if self.has_alignment_heads else None
        tgt = seq[1:]
        keep = np.array([i for i, t in enumerate(tgt) if t < self.eot], dtype=np.int64)
        tok = self.proc.tokenizer
        pieces = [tok.decode([int(seq[p + 1])]) for p in keep]
        text = tok.decode(seq, skip_special_tokens=True).strip()
        tr = Transcript(ids=seq, keep=keep, pieces=pieces, text=text,
                        times=times[keep + 1] if times is not None else None)
        tr.group_words()
        return tr

    def surprisal(self, wavs: np.ndarray, tr: Transcript):
        feats, _ = self.features(np.asarray(wavs, np.float32))
        seq = torch.as_tensor(tr.ids, device=self.device)
        keep = torch.as_tensor(tr.keep, device=self.device)
        dec_in = seq[:-1].unsqueeze(0).expand(feats.shape[0], -1)
        with torch.no_grad():
            logits = self.model(input_features=feats, decoder_input_ids=dec_in).logits.float()
        lp = torch.log_softmax(logits, -1)
        tgt = seq[1:]
        pos = torch.arange(len(tgt), device=self.device)
        s = -lp[:, pos, tgt][:, keep]
        arg = logits[:, keep].argmax(-1)
        return s.cpu().numpy(), arg.cpu().numpy()
