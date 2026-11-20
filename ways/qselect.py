"""Choosing among several speech models' transcripts of one clip.

Every model's transcript is described by four numbers, and a scorer turns them into a predicted
1 - WER. The transcript with the highest score is kept.

  surprisal      the model's mean surprisal of its own transcript (nats per token)
  disagreement   the transcript's mean word error rate against the other models' transcripts
  rate           its length in words relative to the median over the models
  flagged        the share of the clip the degradation detector flags (0.5 when no detector ran)

The scorer is one of the gates of ways/qgate.py (the four-qubit circuit, the same circuit without
entanglement, or the perceptron) plus one learned offset per model. It is fitted by
experiments/qselect_train.py, and the fitted scorers are shipped in ways/gates/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import wer
from .qgate import ClassicalGate, QuantumGate

GATES = Path(__file__).resolve().parent / "gates"
SCORERS = {"quantum": "qselect__quantum.json", "quantum-no-entanglement": "qselect__quantum_no_entanglement.json", "classical": "qselect__classical.json"}


def hypothesis_features(texts: list[str], surprisals: list[float], flagged: float | None) -> np.ndarray:
    """-> [n_models, 4], each feature in [-1, 1]. The scales are fixed, not fitted."""
    n_words = np.array([max(len(t.split()), 1) for t in texts])
    out = []
    for i, text in enumerate(texts):
        others = [wer(o, text)["wer"] if o.strip() else float(bool(text.strip())) for j, o in enumerate(texts) if j != i]
        out.append([np.tanh(surprisals[i] - 0.5), 2 * min(float(np.mean(others)), 1.0) - 1 if others else 0.0,
                    np.tanh(np.log(n_words[i] / np.median(n_words))), 2 * (0.5 if flagged is None else flagged) - 1])
    return np.asarray(out, np.float32)


class Scorer(nn.Module):
    """A gate on the four features plus one offset per model. `models` fixes the order of the offsets."""

    def __init__(self, gate: nn.Module, models: list[str]):
        super().__init__()
        self.gate, self.models = gate, list(models)
        self.offset = nn.Parameter(torch.zeros(len(models)))

    def forward(self, x: torch.Tensor, model_index: torch.Tensor) -> torch.Tensor:
        return self.gate(x) + self.offset[model_index]


def save_scorer(scorer: Scorer, path, **meta) -> None:
    state = {k: v.detach().cpu().tolist() for k, v in scorer.state_dict().items()}
    Path(path).write_text(json.dumps(dict(config=scorer.gate.config(), models=scorer.models, state=state, **meta), indent=1))


def load_scorer(path) -> Scorer:
    d = json.loads(Path(path).read_text())
    c = dict(d["config"]); kind = c.pop("kind")
    scorer = Scorer(QuantumGate(**c) if kind == "quantum" else ClassicalGate(**c), d["models"])
    scorer.load_state_dict({k: torch.tensor(v) for k, v in d["state"].items()})
    return scorer.eval()


def choose(scorer: Scorer, names: list[str], texts: list[str], surprisals: list[float], flagged: float | None):
    """-> (index of the transcript to keep, score per transcript). `names` are the models, in the order of `texts`."""
    missing = [n for n in names if n not in scorer.models]
    if missing:
        raise ValueError(f"this scorer was fitted for {scorer.models} and has no offset for {missing}")
    x = torch.as_tensor(hypothesis_features(texts, surprisals, flagged))
    with torch.no_grad():
        scores = scorer(x, torch.tensor([scorer.models.index(n) for n in names])).numpy()
    return int(scores.argmax()), [float(s) for s in scores]
