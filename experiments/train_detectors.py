"""Train two more degradation detectors with AADD's own recipe, so the noise-focus question can be asked
under detectors whose backbone is not whisper-small (AADD v3.1's):

  whisper-base    AADD's WhisperFrameModel with the openai/whisper-base encoder (their class, their head)
  moonshine-base  the same head on the UsefulSensors/moonshine-base encoder (raw-waveform front end)

Data: LibriSpeech dev-clean (never test-clean, which the controlled set uses), 8 s windows degraded on the
fly by aadd.v3.synth.Synthesizer with AADD's event sampler (0 to 3 degradations per window, random spans),
coloured noise and a crowd of other dev-clean speakers for "noise", synthetic room impulse responses for
"reverb", AADD's own code for the rest. Loss: AADD's masked BCE. Validation: 160 fixed windows, frame AUC.

Usage: python experiments/train_detectors.py whisper-base   [--steps 1200 --bs 12]
       python experiments/train_detectors.py moonshine-base [--steps 1200 --bs 12]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from aadd.v3.synth import CLASSES_V3, Assets, Synthesizer
from common import REVISIONS
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
LS = HERE / "cache/librispeech/LibriSpeech/dev-clean"
OUT = HERE / "results/detectors"
SR, FRAME, SECONDS = 16000, 320, 8.0
N = int(SECONDS * SR)


# ----------------------------------------------------------------------------- data
def load_speech(limit=None):
    files = sorted(LS.rglob("*.flac"))[:limit]
    wavs = []
    for f in files:
        w, sr = sf.read(f, dtype="float32")
        assert sr == SR
        wavs.append((w * 32767).astype(np.int16))
    return files, wavs


def synth_rirs(rng, n=8):
    out = []
    for _ in range(n):
        rt60 = float(rng.uniform(0.3, 1.2)); n_s = int(1.5 * SR); t = np.arange(n_s) / SR
        tail = rng.standard_normal(n_s).astype(np.float32) * np.exp(-6.9078 * t / rt60).astype(np.float32)
        h = np.zeros(n_s, np.float32); h[0] = 1.0
        for _ in range(int(rng.integers(3, 9))):
            h[int(rng.uniform(0.004, 0.06) * SR)] += float(rng.uniform(0.2, 0.7))
        h[int(0.01 * SR):] += float(rng.uniform(0.2, 0.5)) * tail[int(0.01 * SR):]
        out.append(h)
    return out


class SynthWindows(Dataset):
    def __init__(self, wavs, seed, length, babble_pool):
        self.wavs, self.seed, self.length = wavs, seed, length
        rng = np.random.default_rng(seed)
        self.assets = Assets(noises=[], music=[], rirs=synth_rirs(rng), babble=babble_pool)
        self.syn = None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.syn is None:
            self.syn = Synthesizer(self.assets, seconds=SECONDS)
        rng = np.random.default_rng([self.seed, i])
        sp = self.wavs[int(rng.integers(len(self.wavs)))].astype(np.float32) / 32767
        x, y, ig, _ = self.syn.make(sp, rng)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(ig)


# ----------------------------------------------------------------------------- models
class MoonshineFrameModel(nn.Module):
    """AADD's v3 head (weighted sum of all encoder layers, LayerNorm, two convs, sigmoid per class) on the
    Moonshine encoder, which reads the raw waveform; its frames (~24 ms) are resampled to the 20 ms grid."""

    def __init__(self, backbone="UsefulSensors/moonshine-base", n_classes=len(CLASSES_V3), head_dim=256):  # noqa: B008  (a module constant, not a call that varies)
        super().__init__()
        from transformers import MoonshineForConditionalGeneration

        self.encoder = MoonshineForConditionalGeneration.from_pretrained(backbone, revision=REVISIONS.get(backbone)).model.encoder
        self.backbone = backbone
        d = self.encoder.config.hidden_size
        L = self.encoder.config.encoder_num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(L))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Conv1d(d, head_dim, 5, padding=2), nn.GELU(), nn.Dropout(0.1),
                                  nn.Conv1d(head_dim, head_dim, 5, padding=2), nn.GELU())
        self.out = nn.Linear(head_dim, n_classes)
        self.n_frames = N // FRAME

    def forward(self, wav):
        hs = self.encoder(input_values=wav.float(), output_hidden_states=True).hidden_states
        w = torch.softmax(self.layer_w, 0)
        h = sum(w[i] * hs[i] for i in range(len(hs)))
        h = self.norm(h).transpose(1, 2)
        h = self.head(h)                                                   # [B, head_dim, T']
        h = F.interpolate(h, size=self.n_frames, mode="linear", align_corners=False).transpose(1, 2)
        return self.out(h)                                                 # [B, 400, C]


def build(name):
    if name == "whisper-base":
        from aadd.v3.model import WhisperFrameModel
        return WhisperFrameModel(backbone="openai/whisper-base", seconds=SECONDS)
    if name == "moonshine-base":
        return MoonshineFrameModel()
    raise ValueError(name)


def masked_bce(logits, y, ig):
    l = F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="none")
    keep = (~ig).float()
    return (l * keep).sum() / keep.sum().clamp(min=1)


def frame_auc(p, y, ig):
    """Macro AUC over classes with both labels present, frames with ignore=True left out (rank formula)."""
    aucs = []
    for c in range(p.shape[-1]):
        m = ~ig[..., c]
        s, t = p[..., c][m], y[..., c][m]
        if t.min() == t.max():
            continue
        r = s.argsort().argsort().astype(np.float64) + 1
        npos, nneg = (t == 1).sum(), (t == 0).sum()
        aucs.append((r[t == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(aucs)) if aucs else float("nan")


# ----------------------------------------------------------------------------- inference, AADD style
@torch.no_grad()
def frame_probs(model, y, device, overlap=1.0, batch=8):
    """Sliding 8 s windows with 1 s overlap, probabilities averaged where windows overlap, the 0.5 s at
    window edges that are not file edges dropped: the same scheme as aadd.v3.infer.FrameModel."""
    n = len(y); nf = int(np.ceil(n / FRAME)); hop = N - int(overlap * SR)
    starts = list(range(0, max(n - N, 0) + 1, hop))
    if starts[-1] + N < n:
        starts.append(n - N)
    wins = [np.pad(y[s:s + N], (0, N - len(y[s:s + N]))) for s in starts]
    acc = np.zeros((nf + N // FRAME, len(CLASSES_V3))); cnt = np.zeros(len(acc))
    for i in range(0, len(wins), batch):
        x = torch.from_numpy(np.stack(wins[i:i + batch])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float()).cpu().numpy()
        for j, s in enumerate(starts[i:i + batch]):
            f0 = s // FRAME; k = p.shape[1]; lo = 0 if s == 0 else min(25, k)
            acc[f0 + lo:f0 + k] += p[j, lo:k]; cnt[f0 + lo:f0 + k] += 1
    return (acc / np.maximum(cnt, 1)[:, None])[:nf].astype(np.float32)


def load_detector(name, device):
    ck = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=False)
    m = build(ck["arch"]); m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


# ----------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["whisper-base", "moonshine-base"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=8001)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files, wavs = load_speech()
    spk = np.array([f.parts[-3] for f in files])
    speakers = sorted(set(spk)); rng = np.random.default_rng(a.seed)
    val_spk = set(rng.choice(speakers, max(2, len(speakers) // 10), replace=False))
    tr = [w for w, s in zip(wavs, spk, strict=True) if s not in val_spk]; va = [w for w, s in zip(wavs, spk, strict=True) if s in val_spk]
    print(f"{a.name}: {len(files)} utterances, {len(speakers)} speakers, {len(tr)} train / {len(va)} val utterances", flush=True)
    babble = [w.astype(np.float32) / 32767 for w in rng.choice(np.array(tr, dtype=object), 40, replace=False)]
    ds = SynthWindows(tr, a.seed, a.steps * a.bs, babble)
    vds = SynthWindows(va, a.seed + 1, 160, babble)
    vX, vY, vIG = zip(*[vds[i] for i in range(len(vds))], strict=True); vX, vY, vIG = torch.stack(vX), torch.stack(vY).numpy(), torch.stack(vIG).numpy()
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=True)
    model = build(a.name).to(dev)
    backbone = [p for n_, p in model.named_parameters() if n_.startswith("encoder.") and p.requires_grad]
    head = [p for n_, p in model.named_parameters() if not n_.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": head, "lr": a.lr_head}, {"params": backbone, "lr": a.lr}], weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params {n_params:.1f}M, steps {a.steps}, batch {a.bs}, device {dev}", flush=True)

    def evaluate():
        model.eval(); P = []
        with torch.no_grad():
            for i in range(0, len(vX), 8):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    P.append(torch.sigmoid(model(vX[i:i + 8].to(dev)).float()).cpu().numpy())
        model.train(); P = np.concatenate(P)
        return frame_auc(P, vY, vIG), float(masked_bce(torch.from_numpy(np.log(P / (1 - P + 1e-7) + 1e-7)), torch.from_numpy(vY), torch.from_numpy(vIG)))

    hist, best, t0, tl = [], -1.0, time.time(), []
    model.train()
    for step, (x, y, ig) in enumerate(dl, start=1):
        f = step / a.warmup if step < a.warmup else 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))
        for g, b in zip(opt.param_groups, base, strict=True):
            g["lr"] = b * f
        x, y, ig = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), ig.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = masked_bce(model(x), y, ig)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl.append(loss.item())
        if step % 50 == 0:
            print(f"step {step} loss {np.mean(tl[-50:]):.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            auc, vl = evaluate()
            hist.append(dict(step=step, loss=float(np.mean(tl[-a.eval_every:])), val_auc=auc, val_bce=vl))
            mark = ""
            if auc > best:
                best = auc; mark = " *"
                torch.save(dict(state_dict=model.state_dict(), arch=a.name, classes=CLASSES_V3, backbone=model.backbone,
                                meta=dict(step=step, val_auc=auc, val_bce=vl, args=vars(a), n_params_M=n_params,
                                          train_utterances=len(tr), val_utterances=len(va), data="LibriSpeech dev-clean")), OUT / f"{a.name}.pt")
            print(f"EVAL step {step} val frame macro-AUC {auc:.4f} bce {vl:.4f}{mark}", flush=True)
            (OUT / f"{a.name}.log.json").write_text(json.dumps(dict(history=hist, best=best), indent=1))
        if step >= a.steps:
            break
    print(f"done {a.name}: best val AUC {best:.4f} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
