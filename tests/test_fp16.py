"""fp16 (AMP) mode tests — run on CPU with numpy float16 (slow but exact same
code path as GPU fp16 storage): dtype flow, master-weight updates, GradScaler
skip/step behavior, and convergence vs fp32.
"""
import numpy as np
import pytest

import axis
from axis import nn, optim
from axis.tensor import Tensor, set_fp16_mode


@pytest.fixture(autouse=True)
def _fp16_mode():
    set_fp16_mode(True)
    yield
    set_fp16_mode(False)


def _fp16_model():
    axis.manual_seed(0)
    m = nn.Transformer(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                       n_kv_heads=2, mlp_hidden=64, max_seq_len=32)
    for p in m.parameters():
        p.data = p.data.astype(np.float16)
    return m


def test_fp16_forward_dtype_flows():
    m = _fp16_model()
    toks = Tensor(np.array([[1, 5, 9, 2]], dtype=np.int64))
    logits = m.forward(toks)
    assert logits.data.dtype == np.float16          # storage stays fp16
    loss = m.loss(toks, toks)
    assert loss.data.dtype == np.float32            # CE always fp32


def test_fp16_training_converges():
    m = _fp16_model()
    opt = optim.AdamW(m.parameters(), lr=1e-3)
    scaler = optim.GradScaler(init_scale=1024.0)
    seq = np.array([[7, 3, 9, 1, 4, 8, 2, 5]], dtype=np.int64)
    inp, tgt = Tensor(seq[:, :-1]), Tensor(seq[:, 1:])
    first = last = None
    for _ in range(30):
        loss = m.loss(inp, tgt)
        lval = float(loss.data)     # read BEFORE backward (freeing releases
                                    # intermediates; scaled loss makes this one)
        scaler.scale(loss).backward()
        stepped = scaler.step(opt)
        opt.zero_grad()
        if stepped:
            first = first if first is not None else lval
            last = lval
    assert first is not None and last < first, f"fp16 loss did not drop: {first}->{last}"
    # params still stored fp16 after master-weight updates
    assert all(p.data.dtype == np.float16 for p in m.parameters())


def test_gradscaler_skips_nonfinite():
    m = _fp16_model()
    opt = optim.AdamW(m.parameters(), lr=1e-3)
    scaler = optim.GradScaler(init_scale=1024.0)
    p0 = opt.params[0]
    before = p0.data.copy()
    # poison one grad with inf
    for p in opt.params:
        p.grad = np.zeros_like(p.data, dtype=np.float16)
    p0.grad = np.full_like(p0.data, np.inf, dtype=np.float16)
    old_scale = scaler.scale_val
    stepped = scaler.step(opt)
    assert not stepped                                # step skipped
    assert scaler.scale_val < old_scale               # scale backed off
    assert np.array_equal(before, p0.data)            # weights untouched


def test_fp16_matches_fp32_direction():
    """Same seed: fp16 first-step loss ≈ fp32 first-step loss (storage noise only)."""
    set_fp16_mode(False)
    axis.manual_seed(0)
    m32 = nn.Transformer(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                         n_kv_heads=2, mlp_hidden=64, max_seq_len=32)
    seq = np.array([[7, 3, 9, 1, 4, 8, 2, 5]], dtype=np.int64)
    inp, tgt = Tensor(seq[:, :-1]), Tensor(seq[:, 1:])
    l32 = float(m32.loss(inp, tgt).data)

    set_fp16_mode(True)
    m16 = _fp16_model()
    l16 = float(m16.loss(inp, tgt).data)
    assert abs(l32 - l16) / abs(l32) < 0.02, f"fp16 loss far from fp32: {l32} vs {l16}"
