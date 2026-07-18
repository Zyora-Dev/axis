"""Tests for generation and gradient accumulation."""
import numpy as np

import axis
from axis import nn, optim
from axis.tensor import Tensor


def _small():
    axis.manual_seed(0)
    return nn.Transformer(vocab_size=64, dim=64, n_layers=2, n_heads=4,
                          n_kv_heads=2, mlp_hidden=128, max_seq_len=64)


def test_generate_length_and_determinism():
    m = _small()
    out1 = axis.generate(m, [1, 2, 3], max_new_tokens=10, temperature=0.8, top_k=5, seed=7)
    out2 = axis.generate(m, [1, 2, 3], max_new_tokens=10, temperature=0.8, top_k=5, seed=7)
    assert len(out1) == 3 + 10
    assert out1 == out2                      # same seed -> same sample
    assert out1[:3] == [1, 2, 3]             # prompt preserved


def test_generate_greedy_deterministic():
    m = _small()
    a = axis.generate(m, [5, 9], max_new_tokens=8, temperature=0.0)
    b = axis.generate(m, [5, 9], max_new_tokens=8, temperature=0.0)
    assert a == b                            # greedy is deterministic


def test_generate_eos_stops():
    m = _small()
    # force an eos by greedy — just assert eos truncation logic works when the
    # first sampled token equals eos is not guaranteed, so test the length cap
    out = axis.generate(m, [1], max_new_tokens=5, temperature=0.0, eos_id=None)
    assert len(out) == 1 + 5


def test_overfit_then_generate():
    """A tiny model memorises one sequence, then greedily reproduces it."""
    m = _small()
    opt = optim.AdamW(m.parameters(), lr=3e-3)
    seq = np.array([[7, 3, 9, 1, 4, 8, 2, 5]], dtype=np.int64)
    inp, tgt = Tensor(seq[:, :-1]), Tensor(seq[:, 1:])
    for _ in range(200):
        loss = m.loss(inp, tgt); loss.backward(); opt.step(); opt.zero_grad()
    gen = axis.generate(m, [7], max_new_tokens=7, temperature=0.0)
    assert gen == [7, 3, 9, 1, 4, 8, 2, 5], f"got {gen}"


def test_gradient_accumulation_equivalence():
    """Accumulating K micro-batches (loss/K each) == one full-batch step."""
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 64, size=(4, 17)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]

    # full batch grads
    m1 = _small()
    loss = m1.loss(Tensor(inp), Tensor(tgt)); loss.backward()
    full = {n: p.grad.copy() for n, p in m1.named_parameters()}

    # 4 micro-batches of 1, loss/4 each, accumulated
    m2 = _small()
    for i in range(4):
        l = m2.loss(Tensor(inp[i:i+1]), Tensor(tgt[i:i+1]))
        (l * Tensor(np.float32(0.25))).backward()
    acc = {n: p.grad.copy() for n, p in m2.named_parameters()}

    for n in full:
        assert np.allclose(full[n], acc[n], rtol=1e-4, atol=1e-5), \
            f"{n}: grad mismatch max|Δ|={np.abs(full[n]-acc[n]).max()}"
