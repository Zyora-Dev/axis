"""Gradient checkpointing tests: grads must match the non-checkpointed run
exactly, training must work, and eval/no_grad paths must be unaffected.
"""
import numpy as np

import axis
from axis import nn, optim, ops
from axis.tensor import Tensor


def _model(ckpt: bool):
    axis.manual_seed(0)
    m = nn.Transformer(vocab_size=64, dim=32, n_layers=3, n_heads=4,
                       n_kv_heads=2, mlp_hidden=64, max_seq_len=32)
    m.grad_checkpoint = ckpt
    return m


def test_checkpoint_grads_match_exactly():
    seq = np.array([[7, 3, 9, 1, 4, 8, 2, 5]], dtype=np.int64)
    inp, tgt = Tensor(seq[:, :-1]), Tensor(seq[:, 1:])

    m0 = _model(False)
    m0.loss(inp, tgt).backward()
    g0 = {n: p.grad.copy() for n, p in m0.named_parameters() if p.grad is not None}

    m1 = _model(True)
    m1.loss(inp, tgt).backward()
    g1 = {n: p.grad.copy() for n, p in m1.named_parameters() if p.grad is not None}

    assert set(g0) == set(g1), f"param grad coverage differs: {set(g0) ^ set(g1)}"
    for n in g0:
        assert np.allclose(g0[n], g1[n], rtol=1e-5, atol=1e-6), \
            f"{n}: grad mismatch max|Δ|={np.abs(g0[n] - g1[n]).max()}"


def test_checkpoint_training_converges():
    m = _model(True)
    opt = optim.AdamW(m.parameters(), lr=3e-3)
    seq = np.array([[7, 3, 9, 1, 4, 8, 2, 5]], dtype=np.int64)
    inp, tgt = Tensor(seq[:, :-1]), Tensor(seq[:, 1:])
    first = last = None
    for _ in range(40):
        loss = m.loss(inp, tgt)
        lval = float(loss.data)
        loss.backward()
        opt.step(); opt.zero_grad()
        first = first if first is not None else lval
        last = lval
    assert last < first * 0.5, f"ckpt training didn't converge: {first}->{last}"


def test_checkpoint_inactive_in_eval():
    """eval() forward must not use checkpointing (no backward machinery)."""
    m = _model(True).eval()
    toks = Tensor(np.array([[1, 2, 3]], dtype=np.int64))
    with axis.no_grad():
        out = m.forward(toks)
    assert out.shape == (1, 3, 64)


def test_functional_checkpoint_simple():
    """ops.checkpoint on a plain function: grads equal the direct run."""
    axis.manual_seed(1)
    w = Tensor(np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32),
               requires_grad=True)
    x = Tensor(np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32),
               requires_grad=True)

    def fn(a):
        return ops.tanh(ops.matmul(a, w))

    # direct
    y0 = fn(x); y0.sum().backward()
    gw0, gx0 = w.grad.copy(), x.grad.copy()
    w.grad = x.grad = None
    # checkpointed
    y1 = ops.checkpoint(fn, x)
    y1.sum().backward()
    assert np.allclose(gw0, w.grad, rtol=1e-6)
    assert np.allclose(gx0, x.grad, rtol=1e-6)
