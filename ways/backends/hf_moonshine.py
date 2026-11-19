"""Moonshine (Useful Sensors, Hugging Face transformers) white-box backend. Moonshine reads the raw
16 kHz waveform, so the STFT-domain erasure is applied to the waveform and nothing else changes."""
from __future__ import annotations

import numpy as np
import torch

from .base import SR, Backend, Transcript


class MoonshineBackend(Backend):
    name = "hf-moonshine"
    supports_surprisal = True

    def __init__(self, model_id: str = "UsefulSensors/moonshine-base", revision: str | None = None,
                 device: str | None = None):
        from transformers import AutoProcessor, MoonshineForConditionalGeneration

        self.model_id, self.revision = model_id, revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_id, revision=revision).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.bos = int(self.model.generation_config.decoder_start_token_id)
        self.eos = int(self.model.generation_config.eos_token_id)
        self.specials = {self.bos, self.eos}
        self.has_alignment_heads = False

    def inputs(self, wavs: np.ndarray):
        x = self.proc(list(np.asarray(wavs, np.float32)), sampling_rate=SR, return_tensors="pt", padding=True)
        return x.input_values.to(self.device), x.attention_mask.to(self.device)

    def transcribe(self, wav: np.ndarray) -> Transcript:
        iv, am = self.inputs(np.asarray(wav, np.float32)[None])
        with torch.no_grad():
            seq = self.model.generate(iv, attention_mask=am)[0].cpu().numpy()
        tgt = seq[1:]
        keep = np.array([i for i, t in enumerate(tgt) if int(t) not in self.specials], dtype=np.int64)
        tok = self.proc.tokenizer
        raw = tok.convert_ids_to_tokens([int(seq[p + 1]) for p in keep])
        pieces = [r.replace("▁", " ") for r in raw]
        text = tok.decode(seq, skip_special_tokens=True).strip()
        tr = Transcript(ids=seq, keep=keep, pieces=pieces, text=text, times=None)
        tr.group_words()
        return tr

    def surprisal(self, wavs: np.ndarray, tr: Transcript):
        iv, am = self.inputs(np.asarray(wavs, np.float32))
        seq = torch.as_tensor(tr.ids, device=self.device)
        keep = torch.as_tensor(tr.keep, device=self.device)
        dec_in = seq[:-1].unsqueeze(0).expand(iv.shape[0], -1)
        with torch.no_grad():
            logits = self.model(input_values=iv, attention_mask=am, decoder_input_ids=dec_in).logits.float()
        lp = torch.log_softmax(logits, -1)
        tgt = seq[1:]
        pos = torch.arange(len(tgt), device=self.device)
        s = -lp[:, pos, tgt][:, keep]
        arg = logits[:, keep].argmax(-1)
        return s.cpu().numpy(), arg.cpu().numpy()
