"""The fused attention kernel computes the same function as the step-by-step one."""
import math
import os

import numpy as np
import pytest
import torch


def test_fused_kernel_equals_the_written_out_formula():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 6, 50, 64, generator=g, dtype=torch.float64) for _ in range(3))
    weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
    fused = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights @ v, fused, atol=1e-12, rtol=0)


@pytest.mark.gpu
@pytest.mark.slow
def test_whisper_surprisal_is_the_same_under_both_kernels():
    from ways import load_backend

    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        fast = load_backend("hf-whisper", "openai/whisper-tiny")
    except Exception:
        pytest.skip("whisper-tiny weights not cached")
    rng = np.random.default_rng(0)
    t = np.arange(3 * 16000) / 16000
    wavs = np.stack([(0.2 * np.sin(2 * np.pi * f * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32) for f in (180, 260)])
    tr = fast.transcribe(wavs[0])
    if tr.n_tokens == 0:
        pytest.skip("no token to score on the synthetic clip")
    s_fast, a_fast = fast.surprisal(wavs, tr)
    fast.fast_attention = False
    s_eager, a_eager = fast.surprisal(wavs, tr)
    assert fast.model.config._attn_implementation == "eager"
    assert np.abs(s_fast - s_eager).max() < 1e-4 and np.array_equal(a_fast, a_eager)
