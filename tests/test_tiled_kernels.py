"""Tiled-matmul tests: assert the shared-memory tiled path actually executes
on GPU (no silent fallback) and matches NumPy at large + ragged shapes."""
import numpy as np
import pytest

import axis
from axis import accel, ops
from axis.tensor import Tensor

requires_gpu = pytest.mark.skipif(not accel.available(), reason="locomp GPU not available")


@pytest.fixture(autouse=True)
def _gpu():
    accel.enable()
    yield
    accel.disable()


@requires_gpu
def test_tiled_matmul_executes_and_matches():
    """Large aligned shape → tiled kernel. Must NOT fall back and must match."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 64, 96)).astype(np.float32)
    b = rng.standard_normal((2, 96, 80)).astype(np.float32)
    out = accel.matmul(a, b)
    assert out is not None, "tiled matmul silently fell back to NumPy"
    np.testing.assert_allclose(out, a @ b, rtol=1e-3, atol=1e-4)


@requires_gpu
def test_tiled_matmul_ragged_edges():
    """Non-multiple-of-16 dims exercise host padding + slice-back."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 33, 47)).astype(np.float32)
    b = rng.standard_normal((3, 47, 29)).astype(np.float32)
    out = accel.matmul(a, b)
    assert out is not None
    np.testing.assert_allclose(out, a @ b, rtol=1e-3, atol=1e-4)


@requires_gpu
def test_tiled_matmul_2d():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((40, 40)).astype(np.float32)
    b = rng.standard_normal((40, 24)).astype(np.float32)
    out = accel.matmul(a, b)
    assert out is not None
    np.testing.assert_allclose(out, a @ b, rtol=1e-3, atol=1e-4)


@requires_gpu
def test_naive_matmul_small_shapes():
    """Tiny shapes route to the naive kernel — still correct."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((1, 4, 8)).astype(np.float32)
    b = rng.standard_normal((1, 8, 4)).astype(np.float32)
    out = accel.matmul(a, b)
    assert out is not None
    np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-5)


@requires_gpu
def test_tiled_matmul_backward_parity():
    """matmul backward uses GPU matmuls too — gradients must match the NumPy
    reference (finite-difference gradcheck is too noisy against fp32 GPU
    forward, so we compare analytic grads GPU-vs-CPU instead)."""
    rng = np.random.default_rng(7)

    def grads(enabled):
        if enabled:
            accel.enable()
        else:
            accel.disable()
        a = Tensor(rng.standard_normal((20, 32)).astype(np.float32), requires_grad=True)
        b = Tensor(rng.standard_normal((32, 24)).astype(np.float32), requires_grad=True)
        # reset RNG so both runs see identical inputs
        rng2 = np.random.default_rng(7)
        a.data = rng2.standard_normal((20, 32)).astype(np.float32)
        b.data = rng2.standard_normal((32, 24)).astype(np.float32)
        ops.sum(ops.matmul(a, b)).backward()
        return a.grad, b.grad

    ga_gpu, gb_gpu = grads(True)
    ga_cpu, gb_cpu = grads(False)
    accel.enable()
    np.testing.assert_allclose(ga_gpu, ga_cpu, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(gb_gpu, gb_cpu, rtol=1e-3, atol=1e-4)
