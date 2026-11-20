"""Choosing among transcripts: the features, the choice, the shipped scorers and the tool's refusals."""
import numpy as np
import pytest
import torch

from ways.qgate import ClassicalGate
from ways.qselect import GATES, SCORERS, Scorer, choose, hypothesis_features, load_scorer, save_scorer

TEXTS = ["the cat sat on the mat", "the cat sat on a mat", "cat cat cat cat cat cat cat cat cat cat cat cat"]


def test_features_are_bounded_and_mean_what_they_say():
    x = hypothesis_features(TEXTS, [0.2, 0.3, 2.5], 0.75)
    assert x.shape == (3, 4) and np.abs(x).max() <= 1
    assert x[2, 0] > x[0, 0] and x[2, 1] > x[0, 1] and x[2, 2] > 0 > x[0, 2] - 1e-9
    assert np.allclose(x[:, 3], 0.5) and np.allclose(hypothesis_features(TEXTS, [0.2, 0.3, 2.5], None)[:, 3], 0.0)
    assert np.isfinite(hypothesis_features(["", "a b", ""], [0.0, 0.1, 0.0], 0.0)).all()


def test_choose_follows_the_scores_and_refuses_unknown_models(tmp_path):
    gate = ClassicalGate()
    with torch.no_grad():
        for p in gate.parameters():
            p.zero_()
    scorer = Scorer(gate, ["a", "b", "c"])
    with torch.no_grad():
        scorer.offset.copy_(torch.tensor([0.0, 0.3, 0.1]))
    best, scores = choose(scorer, ["c", "b", "a"], TEXTS, [0.2, 0.3, 2.5], 0.5)
    assert best == 1 and np.allclose(scores, [0.6, 0.8, 0.5])
    save_scorer(scorer, tmp_path / "s.json")
    assert choose(load_scorer(tmp_path / "s.json"), ["c", "b", "a"], TEXTS, [0.2, 0.3, 2.5], 0.5)[1] == scores
    with pytest.raises(ValueError):
        choose(scorer, ["a", "z"], TEXTS[:2], [0.1, 0.1], None)


def test_shipped_scorers_load_and_reject_the_looping_transcript():
    names = ["whisper-tiny", "whisper-base", "moonshine-base"]
    for key, file in SCORERS.items():
        if key != "classical":
            pytest.importorskip("pennylane")
        scorer = load_scorer(GATES / file)
        assert scorer.models == names
        best, scores = choose(scorer, names, [TEXTS[2], TEXTS[0], TEXTS[1]], [2.5, 0.2, 0.3], 0.6)
        assert best != 0 and len(scores) == 3


def test_tool_needs_two_models_and_a_known_scorer():
    from ways.tools import q_select

    wav = np.zeros(16000, np.float32)
    with pytest.raises(ValueError):
        q_select(wav, backends=[("hf-whisper", "openai/whisper-tiny")])
    with pytest.raises(ValueError):
        q_select(wav, scorer="nope")
