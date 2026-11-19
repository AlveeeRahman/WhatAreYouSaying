"""The black-box path end to end: start the mock OpenAI-compatible server (tests/mock_server.py), run
the selection-change erasure test through HTTP on both routes, and check that what comes back
agrees with the white-box flips of the same model. Needs the cached whisper-tiny weights."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
HERE = Path(__file__).resolve().parent
CLIP = os.environ.get("WAYS_TEST_CLIP", "$WAYS_DATA/"
                      "audio/id10288_FIxC_PNJNkc_00003.wav")


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def server():
    if not Path(CLIP).exists():
        pytest.skip("test clip not available")
    port = _free_port()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = subprocess.Popen([sys.executable, str(HERE / "mock_server.py"), str(port)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for _ in range(120):
        line = proc.stdout.readline()
        if "mock server on" in line:
            break
        time.sleep(0.5)
    yield f"http://127.0.0.1:{port}/v1"
    proc.terminate()


@pytest.mark.parametrize("route", ["transcriptions", "chat"])
def test_black_box_selection_scoring_matches_white_box_flips(server, route):
    import soundfile as sf
    from ways import explain
    from ways.backends.hf_whisper import WhisperBackend
    from ways.backends.openai_compat import OpenAICompatBackend

    wav, sr = sf.read(CLIP, dtype="float32")
    wav = wav[: 3 * sr]                          # 3 s keeps the number of HTTP calls small
    bb = OpenAICompatBackend(base_url=server, model_id="openai/whisper-tiny", route=route)
    att = explain(bb, wav, patch_s=0.5, n_bands=2, batch=8)
    assert att.mode == "selection" and att.delta.shape[1:] == (2, 6)
    assert att.text == bb.transcribe(wav).text
    assert set(np.unique(att.delta)) <= {0.0, 1.0}
    assert att.delta.sum() > 0, "erasing half-second, half-spectrum patches changed no word at all"
    # the same model white-box: a word whose every token keeps its argmax under erasure of patch P
    # cannot change under greedy re-decoding of P unless an earlier word changed (teacher forcing
    # holds the prefix fixed); so word-level change implies some flip in or before that word
    wb = WhisperBackend("openai/whisper-tiny")
    att_w = explain(wb, wav, patch_s=0.5, n_bands=2, batch=8)
    tr = wb.transcribe(wav)
    words = tr.group_words()
    assert [w["word"] for w in words] == [p.strip() for p in att.pieces]
    for b in range(2):
        for t in range(att.n_t):
            if att.delta[:, b, t].any():
                first = int(np.argmax(att.delta[:, b, t]))
                toks = [k for w in words[: first + 1] for k in w["toks"]]
                assert att_w.flip[toks, b, t].any(), (b, t, first)
