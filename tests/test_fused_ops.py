"""Phase 2.5 fused-op tests: gradcheck on CPU, parity GPU↔CPU, and parity of
the fused attention against the naive composed implementation."""
import math

import numpy as np
import pytest

import axis
from axis import accel, nn, ops
from axis.tensor import Tensor

requires_gpu = pytest.mark.skipif(not accel.available(), reason="locomp GPU not available")


def t(shape, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return Tensor((rng.standard_normal(shape) * scale).astype(np.float32), requires_grad=True)


# ─── silu_mul ───────────────────────────────────────────────────────────────

def test_silu_mul_gradcheck_cpu():
    accel.disable()
    axis.gradcheck(lambda g, u: ops.sum(ops.silu_mul(g, u)), [t((3, 4)), t((3, 4), 1)])


def test_silu_mul_matches_composition():
    accel.disable()
    g, u = t((5, 6), 2), t((5, 6), 3)
    fused = ops.silu_mul(g, u).data
    composed = ops.mul(ops.silu(g), u).data
    np.testing.assert_allclose(fused, composed, rtol=1e-6)


@requires_gpu
def test_silu_mul_gpu_parity():
    g, u = t((100,), 4), t((100,), 5)
    accel.enable()
    gpu = ops.silu_mul(g, u).data
    accel.disable()
    cpu = ops.silu_mul(g, u).data
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-6)


# ─── fused_causal_attention ─────────────────────────────────────────────────

def _naive_attention(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """The original 5-op composition — ground truth for the fused op."""
    T = q.shape[2]
    att = ops.matmul(q, ops.transpose(k, -2, -1))
    att = ops.mul(att, Tensor(np.float32(scale)))
    mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
    att = ops.add(att, Tensor(mask[None, None]))
    att = ops.softmax(att, axis=-1)
    return ops.matmul(att, v)


def test_fused_attention_matches_naive_cpu():
    accel.disable()
    scale = 1.0 / math.sqrt(4)
    q, k, v = t((2, 3, 5, 4), 0, 0.5), t((2, 3, 5, 4), 1, 0.5), t((2, 3, 5, 4), 2, 0.5)
    fused = ops.fused_causal_attention(q, k, v, scale).data
    naive = _naive_attention(q, k, v, scale).data
    np.testing.assert_allclose(fused, naive, rtol=1e-4, atol=1e-5)


def test_fused_attention_gradcheck_cpu():
    accel.disable()
    scale = 1.0 / math.sqrt(3)
    q, k, v = t((1, 2, 4, 3), 3, 0.5), t((1, 2, 4, 3), 4, 0.5), t((1, 2, 4, 3), 5, 0.5)
    axis.gradcheck(
        lambda qq, kk, vv: ops.sum(ops.fused_causal_attention(qq, kk, vv, scale)),
        [q, k, v], rtol=2e-2, atol=2e-3,
    )


def test_fused_attention_grads_match_naive():
    """Backward of the fused op must equal backward of the composition."""
    accel.disable()
    scale = 1.0 / math.sqrt(4)

    def grads(fn):
        q, k, v = t((2, 2, 5, 4), 10, 0.5), t((2, 2, 5, 4), 11, 0.5), t((2, 2, 5, 4), 12, 0.5)
        out = ops.sum(fn(q, k, v))
        out.backward()
        return q.grad, k.grad, v.grad

    fused = grads(lambda q, k, v: ops.fused_causal_attention(q, k, v, scale))
    naive = grads(lambda q, k, v: _naive_attention(q, k, v, scale))
    for f, n in zip(fused, naive):
        np.testing.assert_allclose(f, n, rtol=1e-3, atol=1e-5)


@requires_gpu
def test_fused_attention_gpu_parity():
    scale = 1.0 / math.sqrt(8)
    q, k, v = t((2, 4, 16, 8), 20, 0.5), t((2, 4, 16, 8), 21, 0.5), t((2, 4, 16, 8), 22, 0.5)
    accel.enable()
    gpu = ops.fused_causal_attention(q, k, v, scale).data
    accel.disable()
    cpu = ops.fused_causal_attention(q, k, v, scale).data
    np.testing.assert_allclose(gpu, cpu, rtol=1e-3, atol=1e-4)


@requires_gpu
def test_fused_attention_gpu_gradcheck():
    accel.enable()
    scale = 1.0 / math.sqrt(3)
    q, k, v = t((1, 2, 4, 3), 30, 0.5), t((1, 2, 4, 3), 31, 0.5), t((1, 2, 4, 3), 32, 0.5)
    try:
        axis.gradcheck(
            lambda qq, kk, vv: ops.sum(ops.fused_causal_attention(qq, kk, vv, scale)),
            [q, k, v], rtol=2e-2, atol=2e-3,
        )
    finally:
        accel.disable()


# ─── module-level: attention output identical accel on/off ──────────────────

@requires_gpu
def test_attention_module_gpu_parity():
    axis.manual_seed(0)
    attn = nn.CausalSelfAttention(dim=16, n_heads=4, n_kv_heads=2, max_seq_len=8)
    x = Tensor(np.random.default_rng(0).standard_normal((2, 6, 16)).astype(np.float32) * 0.5)
    accel.enable()
    gpu = attn(x).data
    accel.disable()
    cpu = attn(x).data
    np.testing.assert_allclose(gpu, cpu, rtol=1e-3, atol=1e-4)
