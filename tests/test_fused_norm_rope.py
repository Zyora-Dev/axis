"""Gradcheck + equivalence tests for the fused rmsnorm and rope ops."""
import numpy as np

import axis
from axis import nn, ops
from axis.gradcheck import gradcheck
from axis.tensor import Tensor


def test_rmsnorm_gradcheck():
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((2, 3, 8)).astype(np.float32), requires_grad=True)
    w = Tensor(rng.standard_normal(8).astype(np.float32), requires_grad=True)
    assert gradcheck(lambda a, b: ops.rmsnorm(a, b, eps=1e-5).sum(), (x, w))


def test_rmsnorm_matches_unfused():
    rng = np.random.default_rng(1)
    xd = rng.standard_normal((2, 5, 16)).astype(np.float32)
    wd = rng.standard_normal(16).astype(np.float32)
    x, w = Tensor(xd), Tensor(wd)
    fused = ops.rmsnorm(x, w, eps=1e-5).numpy()
    ms = (xd * xd).mean(axis=-1, keepdims=True)
    ref = xd / np.sqrt(ms + 1e-5) * wd
    assert np.allclose(fused, ref, rtol=1e-5, atol=1e-6)


def test_rope_gradcheck():
    rng = np.random.default_rng(2)
    T, D = 4, 8
    cos, sin = nn._rope_cache(T, D)
    x = Tensor(rng.standard_normal((1, 2, T, D)).astype(np.float32), requires_grad=True)
    assert gradcheck(lambda a: ops.rope(a, cos, sin).sum(), (x,))


def test_rope_rotation_is_orthogonal():
    """RoPE preserves the norm of each (x1_i, x2_i) pair — sanity of the math."""
    rng = np.random.default_rng(3)
    T, D = 6, 8
    cos, sin = nn._rope_cache(T, D)
    xd = rng.standard_normal((1, 1, T, D)).astype(np.float32)
    out = ops.rope(Tensor(xd), cos, sin).numpy()
    h = D // 2
    n_in = xd[..., :h] ** 2 + xd[..., h:] ** 2
    n_out = out[..., :h] ** 2 + out[..., h:] ** 2
    assert np.allclose(n_in, n_out, rtol=1e-4, atol=1e-5)
