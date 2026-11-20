"""A learned gate on audio patches, placed in front of a speech-to-text model.

The erasure test asks what happens when one patch of the audio is removed. The gate turns that
around: for every patch of the same grid (100 ms x 8 bands) it outputs a number g in [0, 1], the
patch's STFT coefficients are multiplied by g, and the waveform is resynthesised and given to the
model. g = 1 keeps the patch and g = 0 erases it. The gate is trained so that the frozen speech
model becomes more confident in the correct words.

Each patch is described by four numbers: its loudness relative to the clip, its loudness relative
to the other bands at that moment, its loudness relative to its neighbours in time, and the share
of its 20 ms frames that the degradation detector flags. Three gates read those four numbers:

  QuantumGate          a four-qubit PennyLane circuit, one qubit per feature. Each layer encodes the
                       features as RX rotations with trainable scales, applies a trainable rotation
                       to every qubit, and entangles neighbours with a ring of CNOTs. The gate value
                       is (1 + mean_i <Z_i>) / 2.
  QuantumGate(entangle=False)   the same circuit without the CNOTs, so no qubit sees another.
  ClassicalGate        a small multilayer perceptron with about the same number of parameters.

The circuit is simulated exactly (default.qubit, state vector). The two controls exist so that a
gain can be attributed: to gating at all, to the circuit, or to entanglement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import erasure

N_FEATURES = 4
WHISPER_SAMPLES = 480000


def _maps(grid: erasure.PatchGrid, n_frames: int, device):
    """Band of every STFT bin and patch of every STFT frame, as index tensors."""
    band = torch.zeros(erasure.N_BINS, dtype=torch.long, device=device)
    for b, (lo, hi) in enumerate(grid.bins):
        band[lo:hi] = b
    patch = torch.clamp(torch.arange(n_frames, device=device) // grid.frames_per_patch, max=grid.n_t - 1)
    return band, patch


def patch_features(wav: torch.Tensor, grid: erasure.PatchGrid, flags: np.ndarray | None = None) -> torch.Tensor:
    """wav [N] -> features [n_bands, n_t, 4], each roughly in [-1, 1]. `flags` is the detector's
    flagged share per patch in time, [n_t] in [0, 1], or None when no detector ran."""
    with torch.no_grad():
        power = erasure.stft(wav).abs() ** 2
        band, patch = _maps(grid, power.shape[1], wav.device)
        to_band = torch.nn.functional.one_hot(band, grid.n_bands).float()
        to_patch = torch.nn.functional.one_hot(patch, grid.n_t).float()
        energy = (to_band / to_band.sum(0)).T @ power @ (to_patch / to_patch.sum(0))
        e = torch.log10(energy + 1e-10)
        spread = e.std() + 1e-6
        loud = (e - e.median()) / spread
        across_bands = (e - e.mean(0, keepdim=True)) / spread
        padded = torch.nn.functional.pad(e[None], (1, 1), mode="replicate")[0]
        across_time = (e - 0.5 * (padded[:, :-2] + padded[:, 2:])) / spread
        f = torch.zeros(grid.n_t, device=wav.device) if flags is None else torch.as_tensor(np.asarray(flags, np.float32), device=wav.device)
        flagged = (2 * f - 1 if flags is not None else f)[None].expand(grid.n_bands, -1)
        return torch.stack([torch.tanh(loud), torch.tanh(across_bands), torch.tanh(across_time), flagged], -1)


class QuantumGate(nn.Module):
    """Four qubits, `n_layers` layers of feature encoding, trainable rotations and optional CNOT ring."""

    kind = "quantum"

    def __init__(self, n_layers: int = 2, entangle: bool = True, seed: int = 8001):
        super().__init__()
        import pennylane as qml

        g = torch.Generator().manual_seed(seed)
        self.n_layers, self.entangle = n_layers, entangle
        self.scale = nn.Parameter(0.1 * torch.randn(n_layers, N_FEATURES, generator=g))
        self.rot = nn.Parameter(0.1 * torch.randn(n_layers, N_FEATURES, 3, generator=g))
        dev = qml.device("default.qubit", wires=N_FEATURES)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, scale, rot):
            for layer in range(n_layers):
                for i in range(N_FEATURES):
                    qml.RX(scale[layer, i] * np.pi * x[:, i], wires=i)
                for i in range(N_FEATURES):
                    qml.Rot(rot[layer, i, 0], rot[layer, i, 1], rot[layer, i, 2], wires=i)
                if entangle:
                    for i in range(N_FEATURES):
                        qml.CNOT(wires=[i, (i + 1) % N_FEATURES])
            return [qml.expval(qml.PauliZ(i)) for i in range(N_FEATURES)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., 4] -> g [...] in [0, 1]. The circuit runs on the CPU, in double precision."""
        flat = x.reshape(-1, N_FEATURES).double().cpu()
        z = torch.stack(list(self.circuit(flat, self.scale.double().cpu(), self.rot.double().cpu())), -1)
        return (0.5 * (1 + z.mean(-1))).float().to(x.device).reshape(x.shape[:-1])

    def config(self) -> dict:
        return dict(kind=self.kind, n_layers=self.n_layers, entangle=self.entangle)


class ClassicalGate(nn.Module):
    """The control: a perceptron with one hidden layer on the same four features."""

    kind = "classical"

    def __init__(self, hidden: int = 6, seed: int = 8001):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.net = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.net[2].weight.mul_(0.1)
            self.net[2].bias.fill_(3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.net[0].weight.device
        return torch.sigmoid(self.net(x.float().to(device))).squeeze(-1).to(x.device)

    def config(self) -> dict:
        return dict(kind=self.kind, hidden=self.hidden)


def n_parameters(gate: nn.Module) -> int:
    return int(sum(p.numel() for p in gate.parameters()))


def save_gate(gate: nn.Module, path, **meta) -> None:
    """Weights as plain JSON: a gate has a few dozen numbers."""
    state = {k: v.detach().cpu().tolist() for k, v in gate.state_dict().items()}
    Path(path).write_text(json.dumps(dict(config=gate.config(), state=state, **meta), indent=1))


def load_gate(path) -> nn.Module:
    d = json.loads(Path(path).read_text())
    c = dict(d["config"]); kind = c.pop("kind")
    gate = QuantumGate(**c) if kind == "quantum" else ClassicalGate(**c)
    gate.load_state_dict({k: torch.tensor(v) for k, v in d["state"].items()})
    return gate.eval()


def apply_gate(wav: torch.Tensor, grid: erasure.PatchGrid, g: torch.Tensor) -> torch.Tensor:
    """Multiply every patch of the STFT by its gate value and resynthesise. wav [N], g [n_bands, n_t]."""
    X = erasure.stft(wav)
    band, patch = _maps(grid, X.shape[1], wav.device)
    return erasure.istft(X * g[band][:, patch], grid.n_samples)


def gate_wav(wav: torch.Tensor, gate: nn.Module, flags: np.ndarray | None = None, patch_s: float = 0.1, n_bands: int = 8):
    """-> (gated waveform [N], gate values [n_bands, n_t]). Differentiable in the gate's parameters."""
    grid = erasure.PatchGrid(len(wav), patch_s, n_bands)
    g = gate(patch_features(wav, grid, flags))
    return apply_gate(wav, grid, g), g


def whisper_features(backend, wav: torch.Tensor) -> torch.Tensor:
    """Whisper's log-mel input computed in torch, so gradients reach the waveform. wav [N] -> [80, 3000].
    Equal to the processor's feature extractor (tests/test_qgate.py)."""
    x = torch.zeros(WHISPER_SAMPLES, device=wav.device, dtype=wav.dtype)
    x[: len(wav)] = wav
    X = torch.stft(x, erasure.N_FFT, erasure.HOP, window=torch.hann_window(erasure.N_FFT, device=wav.device), return_complex=True)
    filters = torch.as_tensor(backend.proc.feature_extractor.mel_filters, device=wav.device, dtype=torch.float32)
    log_spec = torch.clamp(filters.T @ (X[:, :-1].abs() ** 2), min=1e-10).log10()
    return (torch.maximum(log_spec, log_spec.max().detach() - 8.0) + 4.0) / 4.0


def reference_ids(backend, text: str) -> torch.Tensor:
    """The decoder sequence Whisper would have to write for a reference text: prefix, text, end."""
    tok = backend.proc.tokenizer
    tok.set_prefix_tokens(language=backend.language, task="transcribe")
    return torch.as_tensor(tok(" " + text.strip()).input_ids, device=backend.device)


def reference_nll(backend, feats: torch.Tensor, ids: list[torch.Tensor]) -> torch.Tensor:
    """Mean surprisal, in nats per token, of each clip's reference text under teacher forcing.
    feats [B, 80, 3000], ids one sequence per clip -> [B]. The four prefix tokens are not scored."""
    longest = max(len(s) for s in ids)
    seq = torch.full((len(ids), longest), backend.eot, device=backend.device, dtype=torch.long)
    mask = torch.zeros((len(ids), longest - 1), device=backend.device)
    for i, s in enumerate(ids):
        seq[i, : len(s)] = s
        mask[i, 3: len(s) - 1] = 1
    with backend._attention("sdpa"):
        logits = backend.model(input_features=feats, decoder_input_ids=seq[:, :-1]).logits.float()
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq[:, 1:], reduction="none")
    return (nll * mask).sum(1) / mask.sum(1)
