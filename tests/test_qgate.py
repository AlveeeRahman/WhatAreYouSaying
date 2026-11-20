"""The patch gate: the circuit against a hand calculation, the gate as a generalised erasure, and the weights file."""
import numpy as np
import pytest
import torch

from ways import erasure

qml = pytest.importorskip("pennylane")
from ways.qgate import ClassicalGate, QuantumGate, apply_gate, gate_wav, load_gate, n_parameters, patch_features, save_gate  # noqa: E402


def _wav(n=16000, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    return torch.as_tensor((0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.standard_normal(n)).astype(np.float32))


def test_one_layer_without_entanglement_matches_the_closed_form():
    gate = QuantumGate(n_layers=1, entangle=False)
    with torch.no_grad():
        gate.rot.zero_()
    x = torch.tensor([[0.2, -0.5, 0.9, 0.0], [1.0, 1.0, -1.0, 0.3]])
    expected = 0.5 * (1 + torch.cos(gate.scale[0].detach() * np.pi * x).mean(-1))
    assert torch.allclose(gate(x), expected, atol=1e-6)


def test_entanglement_changes_the_output_and_gradients_flow():
    x = torch.tensor([[0.2, -0.5, 0.9, 0.4]])
    a, b = QuantumGate(entangle=True), QuantumGate(entangle=False)
    assert abs(float(a(x)) - float(b(x))) > 1e-4
    a(x).sum().backward()
    assert a.scale.grad.abs().sum() > 0 and a.rot.grad.abs().sum() > 0
    assert n_parameters(a) == 32 and 0 <= float(a(x)) <= 1


def test_gate_of_one_keeps_the_audio_and_zero_is_one_erasure_step():
    wav = _wav()
    grid = erasure.PatchGrid(len(wav), 0.1, 8)
    ones = torch.ones(8, grid.n_t)
    assert (apply_gate(wav, grid, ones) - wav).abs().max() < 1e-5
    g = ones.clone(); g[1, 4] = 0
    assert (apply_gate(wav, grid, g) - erasure.erase(wav, grid, [(1, 4)], n_iter=1)[0]).abs().max() < 1e-6


def test_features_shape_range_and_detector_column():
    wav = _wav()
    grid = erasure.PatchGrid(len(wav), 0.1, 8)
    flags = np.r_[np.ones(5), np.zeros(5)]
    f = patch_features(wav, grid, flags)
    assert f.shape == (8, 10, 4) and f.abs().max() <= 1
    assert (f[:, :5, 3] == 1).all() and (f[:, 5:, 3] == -1).all() and (patch_features(wav, grid)[..., 3] == 0).all()
    assert f[1, :, 0].mean() > f[6, :, 0].mean()


def test_weights_round_trip_and_gated_audio(tmp_path):
    wav = _wav()
    for gate in (QuantumGate(), QuantumGate(entangle=False), ClassicalGate()):
        save_gate(gate, tmp_path / "g.json", model="m")
        again = load_gate(tmp_path / "g.json")
        y1, g1 = gate_wav(wav, gate)
        y2, g2 = gate_wav(wav, again)
        assert torch.allclose(g1, g2, atol=1e-6) and torch.allclose(y1, y2, atol=1e-6) and g1.shape == (8, 10)
        assert float(g1.min()) >= 0 and float(g1.max()) <= 1
