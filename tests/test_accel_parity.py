"""GPU ↔ NumPy parity tests.

The Phase 2 contract: with acceleration enabled, every op must produce the
same result as the NumPy reference within float32 tolerance — including a
full transformer training step. Skipped automatically when no GPU/locomp.
"""
import numpy as np
import pytest

import axis
from axis import accel, nn, ops, optim
from axis.tensor import Tensor

requires_gpu = pytest.mark.skipif(not accel.available(), reason="locomp GPU not available")


@pytest.fixture(autouse=True)
def _gpu_on_off():
    accel.enable()
    yield
    accel.disable()


def _ref(fn, *arrays):
    """Run fn with acceleration disabled → NumPy ground truth."""
    accel.disable()
    try:
        return fn(*arrays)
    finally:
        accel.enable()


# ─── op-level parity ────────────────────────────────────────────────────────


@requires_gpu
def test_matmul_2d_parity():
    rng = np.random.default_rng(0)
    a = Tensor(rng.standard_normal((16, 32)).astype(np.float32))
    b = Tensor(rng.standard_normal((32, 24)).astype(np.float32))
    gpu = ops.matmul(a, b).data
    ref = _ref(lambda: ops.matmul(a, b).data)
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-5)


@requires_gpu
def test_matmul_batched_parity():
    rng = np.random.default_rng(1)
    a = Tensor(rng.standard_normal((4, 8, 16)).astype(np.float32))
    b = Tensor(rng.standard_normal((4, 16, 12)).astype(np.float32))
    gpu = ops.matmul(a, b).data
    ref = _ref(lambda: ops.matmul(a, b).data)
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-5)


@requires_gpu
def test_softmax_parity():
    rng = np.random.default_rng(2)
    x = Tensor(rng.standard_normal((32, 64)).astype(np.float32) * 4.0)
    gpu = ops.softmax(x).data
    ref = _ref(lambda: ops.softmax(x).data)
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(gpu.sum(-1), 1.0, rtol=1e-5)


@requires_gpu
def test_silu_parity():
    rng = np.random.default_rng(3)
    x = Tensor(rng.standard_normal((1000,)).astype(np.float32) * 3.0)
    gpu = ops.silu(x).data
    ref = _ref(lambda: ops.silu(x).data)
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-6)


@requires_gpu
def test_gelu_parity():
    rng = np.random.default_rng(4)
    x = Tensor(rng.standard_normal((1000,)).astype(np.float32) * 3.0)
    gpu = ops.gelu(x).data
    ref = _ref(lambda: ops.gelu(x).data)
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-6)


# ─── gradients still correct with GPU forward ───────────────────────────────


@requires_gpu
def test_matmul_gradcheck_on_gpu():
    rng = np.random.default_rng(5)
    a = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
    b = Tensor(rng.standard_normal((4, 5)).astype(np.float32), requires_grad=True)
    axis.gradcheck(lambda x, y: ops.sum(ops.matmul(x, y)), [a, b])


# ─── end-to-end: full training-step parity ──────────────────────────────────


@requires_gpu
def test_transformer_loss_parity():
    """Same seed, same data → GPU loss must match NumPy loss closely."""
    def build_and_loss():
        axis.manual_seed(7)
        model = nn.Transformer(vocab_size=32, dim=16, n_layers=2, n_heads=4,
                               n_kv_heads=2, mlp_hidden=32, max_seq_len=8)
        tokens = Tensor(np.arange(8, dtype=np.int64)[None] % 32)
        targets = Tensor((np.arange(8, dtype=np.int64)[None] + 1) % 32)
        return model.loss(tokens, targets).item()

    accel.enable()
    loss_gpu = build_and_loss()
    accel.disable()
    loss_cpu = build_and_loss()
    assert abs(loss_gpu - loss_cpu) < 1e-3, f"gpu={loss_gpu} cpu={loss_cpu}"


@requires_gpu
def test_training_convergence_on_gpu():
    """The overfit proof must also hold with GPU kernels in the loop."""
    axis.manual_seed(99)
    vocab, seq = 16, 8
    model = nn.Transformer(vocab_size=vocab, dim=32, n_layers=2, n_heads=4,
                           n_kv_heads=2, mlp_hidden=64, max_seq_len=seq)
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    rng = np.random.default_rng(0)
    tokens_np = rng.integers(0, vocab, size=(4, seq)).astype(np.int64)
    inputs, targets = Tensor(tokens_np[:, :-1]), Tensor(tokens_np[:, 1:])

    loss_val = None
    for _ in range(150):
        model.zero_grad()
        loss = model.loss(inputs, targets)
        loss.backward()
        optim.clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        loss_val = loss.item()
    assert loss_val is not None and loss_val < 0.5, f"GPU training failed to converge: {loss_val}"


# ─── graceful degradation ───────────────────────────────────────────────────


def test_disabled_accel_is_pure_numpy():
    accel.disable()
    rng = np.random.default_rng(0)
    a = Tensor(rng.standard_normal((4, 4)).astype(np.float32))
    b = Tensor(rng.standard_normal((4, 4)).astype(np.float32))
    out = ops.matmul(a, b).data
    np.testing.assert_allclose(out, a.data @ b.data, rtol=1e-6)
