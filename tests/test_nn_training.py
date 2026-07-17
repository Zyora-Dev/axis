"""Module-level tests: layers gradcheck, determinism, checkpoint round-trip,
and the end-to-end proof — a tiny transformer that actually learns."""
import os
import tempfile

import numpy as np
import pytest

import axis
from axis import nn, ops, optim
from axis.tensor import Tensor


# ─── layers gradcheck (through composed modules) ────────────────────────────

def test_linear_grad():
    axis.manual_seed(0)
    lin = nn.Linear(4, 3)
    x = Tensor(np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32), requires_grad=True)
    axis.gradcheck(lambda w, b, xx: ops.sum(ops.add(ops.matmul(xx, w), b)),
                   [lin.weight, lin.bias, x])


def test_rmsnorm_grad():
    axis.manual_seed(0)
    norm = nn.RMSNorm(6)
    x = Tensor(np.random.default_rng(1).standard_normal((3, 6)).astype(np.float32), requires_grad=True)
    def f(w, xx):
        norm.weight.data = w.data  # not needed; use module directly
        return ops.sum(norm(xx))
    axis.gradcheck(lambda xx: ops.sum(norm(xx)), [x])


def test_layernorm_grad():
    axis.manual_seed(0)
    norm = nn.LayerNorm(6)
    x = Tensor(np.random.default_rng(2).standard_normal((3, 6)).astype(np.float32), requires_grad=True)
    axis.gradcheck(lambda xx: ops.sum(norm(xx)), [x])


def test_swiglu_grad():
    axis.manual_seed(0)
    mlp = nn.SwiGLU(4, 8)
    x = Tensor(np.random.default_rng(3).standard_normal((2, 4)).astype(np.float32), requires_grad=True)
    axis.gradcheck(lambda xx: ops.sum(mlp(xx)), [x])


def test_attention_grad():
    axis.manual_seed(0)
    attn = nn.CausalSelfAttention(dim=8, n_heads=2, n_kv_heads=1, max_seq_len=8)
    x = Tensor(np.random.default_rng(4).standard_normal((1, 4, 8)).astype(np.float32) * 0.5,
               requires_grad=True)
    axis.gradcheck(lambda xx: ops.sum(attn(xx)), [x], rtol=2e-2, atol=2e-3)


# ─── determinism ────────────────────────────────────────────────────────────

def test_deterministic_init_and_forward():
    def build_and_run():
        axis.manual_seed(42)
        model = nn.Transformer(vocab_size=17, dim=8, n_layers=2, n_heads=2, max_seq_len=8)
        tokens = Tensor(np.array([[1, 2, 3, 4]], dtype=np.int64))
        return model(tokens).data.copy()
    a = build_and_run()
    b = build_and_run()
    np.testing.assert_array_equal(a, b)  # bit-identical


# ─── module system ──────────────────────────────────────────────────────────

def test_state_dict_roundtrip():
    axis.manual_seed(0)
    m1 = nn.Transformer(vocab_size=11, dim=8, n_layers=2, n_heads=2, max_seq_len=8)
    axis.manual_seed(99)
    m2 = nn.Transformer(vocab_size=11, dim=8, n_layers=2, n_heads=2, max_seq_len=8)
    m2.load_state_dict(m1.state_dict())
    tokens = Tensor(np.array([[1, 2, 3]], dtype=np.int64))
    np.testing.assert_array_equal(m1(tokens).data, m2(tokens).data)


def test_num_parameters_counts():
    axis.manual_seed(0)
    m = nn.Transformer(vocab_size=10, dim=8, n_layers=1, n_heads=2, mlp_hidden=16, max_seq_len=8)
    assert m.num_parameters() > 0
    assert len(list(m.named_parameters())) == len(list(m.parameters()))


# ─── checkpoint ─────────────────────────────────────────────────────────────

def test_checkpoint_atomic_roundtrip():
    axis.manual_seed(0)
    model = nn.Transformer(vocab_size=13, dim=8, n_layers=1, n_heads=2, max_seq_len=8)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    sched = optim.CosineWithWarmup(opt, warmup_steps=2, max_steps=10, max_lr=1e-3)

    # One training step so optimizer state is non-trivial.
    tokens = Tensor(np.array([[1, 2, 3, 4]], dtype=np.int64))
    targets = Tensor(np.array([[2, 3, 4, 5]], dtype=np.int64))
    loss = model.loss(tokens, targets)
    loss.backward()
    sched.step()
    opt.step()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.npz")
        axis.save(path, model=model, optimizer=opt, scheduler=sched, step=1,
                  meta={"note": "test"})

        axis.manual_seed(7)
        model2 = nn.Transformer(vocab_size=13, dim=8, n_layers=1, n_heads=2, max_seq_len=8)
        opt2 = optim.AdamW(model2.parameters(), lr=1e-3)
        sched2 = optim.CosineWithWarmup(opt2, warmup_steps=2, max_steps=10, max_lr=1e-3)
        header = axis.load(path, model=model2, optimizer=opt2, scheduler=sched2)

        assert header["step"] == 1
        assert header["meta"]["note"] == "test"
        assert sched2.step_num == sched.step_num
        assert opt2.t == opt.t
        np.testing.assert_array_equal(model(tokens).data, model2(tokens).data)


# ─── THE proof: a transformer that learns ───────────────────────────────────

def test_transformer_overfits_copy_task():
    """Tiny transformer must drive loss near zero on a fixed sequence-copy task.
    This is the end-to-end guarantee: autograd + attention + optimizer all
    correct together."""
    axis.manual_seed(1234)
    vocab, seq = 16, 8
    model = nn.Transformer(vocab_size=vocab, dim=32, n_layers=2, n_heads=4,
                           n_kv_heads=2, mlp_hidden=64, max_seq_len=seq)
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    rng = np.random.default_rng(0)
    tokens_np = rng.integers(0, vocab, size=(4, seq)).astype(np.int64)
    # Next-token prediction on fixed data (memorization test).
    inputs = Tensor(tokens_np[:, :-1])
    targets = Tensor(tokens_np[:, 1:])

    first_loss = None
    loss_val = None
    for step in range(300):
        model.zero_grad()
        loss = model.loss(inputs, targets)
        loss.backward()
        optim.clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        loss_val = loss.item()
        if first_loss is None:
            first_loss = loss_val

    assert first_loss is not None and loss_val is not None
    assert loss_val < 0.1, f"transformer failed to memorize: loss={loss_val:.4f} (start {first_loss:.4f})"
    assert loss_val < first_loss / 10.0


def test_gqa_matches_mha_shapes():
    axis.manual_seed(0)
    m = nn.CausalSelfAttention(dim=16, n_heads=4, n_kv_heads=2, max_seq_len=8)
    x = Tensor(np.random.default_rng(0).standard_normal((2, 5, 16)).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 5, 16)
