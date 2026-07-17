"""Gradcheck every Axis op against central finite differences."""
import numpy as np
import pytest

import axis
from axis import ops
from axis.tensor import Tensor


def t(shape, seed=0, scale=1.0, positive=False):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32) * scale
    if positive:
        data = np.abs(data) + 0.5
    return Tensor(data, requires_grad=True)


# ─── elementwise binary ─────────────────────────────────────────────────────

def test_add():        axis.gradcheck(lambda a, b: ops.sum(ops.add(a, b)), [t((3, 4)), t((3, 4), 1)])
def test_add_broadcast(): axis.gradcheck(lambda a, b: ops.sum(ops.add(a, b)), [t((3, 4)), t((4,), 1)])
def test_sub():        axis.gradcheck(lambda a, b: ops.sum(ops.sub(a, b)), [t((3, 4)), t((3, 4), 1)])
def test_mul():        axis.gradcheck(lambda a, b: ops.sum(ops.mul(a, b)), [t((3, 4)), t((3, 4), 1)])
def test_mul_broadcast(): axis.gradcheck(lambda a, b: ops.sum(ops.mul(a, b)), [t((2, 3, 4)), t((4,), 1)])
def test_div():        axis.gradcheck(lambda a, b: ops.sum(ops.div(a, b)), [t((3, 4)), t((3, 4), 1, positive=True)])
def test_pow():        axis.gradcheck(lambda a: ops.sum(ops.pow(a, 3.0)), [t((3, 4))])
def test_maximum():    axis.gradcheck(lambda a, b: ops.sum(ops.maximum(a, b)), [t((3, 4)), t((3, 4), 7)])


# ─── elementwise unary ──────────────────────────────────────────────────────

def test_exp():     axis.gradcheck(lambda a: ops.sum(ops.exp(a)), [t((3, 4), scale=0.5)])
def test_log():     axis.gradcheck(lambda a: ops.sum(ops.log(a)), [t((3, 4), positive=True)])
def test_sqrt():    axis.gradcheck(lambda a: ops.sum(ops.sqrt(a)), [t((3, 4), positive=True)])
def test_tanh():    axis.gradcheck(lambda a: ops.sum(ops.tanh(a)), [t((3, 4))])
def test_sigmoid(): axis.gradcheck(lambda a: ops.sum(ops.sigmoid(a)), [t((3, 4))])
def test_relu():    axis.gradcheck(lambda a: ops.sum(ops.relu(a)), [t((3, 4), scale=2.0)])
def test_gelu():    axis.gradcheck(lambda a: ops.sum(ops.gelu(a)), [t((3, 4))])
def test_silu():    axis.gradcheck(lambda a: ops.sum(ops.silu(a)), [t((3, 4))])


# ─── matmul ─────────────────────────────────────────────────────────────────

def test_matmul_2d():      axis.gradcheck(lambda a, b: ops.sum(ops.matmul(a, b)), [t((3, 4)), t((4, 5), 1)])
def test_matmul_batched():  axis.gradcheck(lambda a, b: ops.sum(ops.matmul(a, b)), [t((2, 3, 4)), t((2, 4, 5), 1)])
def test_matmul_broadcast(): axis.gradcheck(lambda a, b: ops.sum(ops.matmul(a, b)), [t((2, 3, 4)), t((4, 5), 1)])


# ─── reductions ─────────────────────────────────────────────────────────────

def test_sum_all():      axis.gradcheck(lambda a: ops.sum(a), [t((3, 4))])
def test_sum_axis():     axis.gradcheck(lambda a: ops.sum(ops.sum(a, axis=1)), [t((3, 4))])
def test_mean_all():     axis.gradcheck(lambda a: ops.mean(a), [t((3, 4))])
def test_mean_axis():    axis.gradcheck(lambda a: ops.sum(ops.mean(a, axis=-1, keepdims=True)), [t((3, 4))])
def test_max():          axis.gradcheck(lambda a: ops.sum(ops.max(a, axis=1)), [t((3, 4))])


# ─── shape ──────────────────────────────────────────────────────────────────

def test_reshape():   axis.gradcheck(lambda a: ops.sum(ops.reshape(a, (4, 3))), [t((3, 4))])
def test_transpose(): axis.gradcheck(lambda a: ops.sum(ops.transpose(a, 0, 1)), [t((3, 4))])
def test_getitem():   axis.gradcheck(lambda a: ops.sum(ops.getitem(a, (slice(0, 2), slice(1, 3)))), [t((3, 4))])
def test_cat():       axis.gradcheck(lambda a, b: ops.sum(ops.cat([a, b], axis=1)), [t((3, 2)), t((3, 3), 1)])


# ─── softmax family ─────────────────────────────────────────────────────────

def test_softmax():     axis.gradcheck(lambda a: ops.sum(ops.mul(ops.softmax(a), Tensor(np.arange(4, dtype=np.float32)))), [t((3, 4))])
def test_log_softmax(): axis.gradcheck(lambda a: ops.sum(ops.mul(ops.log_softmax(a), Tensor(np.arange(4, dtype=np.float32)))), [t((3, 4))])


# ─── embedding + cross-entropy ──────────────────────────────────────────────

def test_embedding():
    idx = Tensor(np.array([[0, 2], [1, 0]], dtype=np.int64))
    axis.gradcheck(lambda w: ops.sum(ops.embedding(w, idx)), [t((4, 3))])


def test_cross_entropy():
    targets = Tensor(np.array([1, 3, 0], dtype=np.int64))
    axis.gradcheck(lambda x: ops.cross_entropy(x, targets), [t((3, 4))])


def test_cross_entropy_ignore_index():
    targets = Tensor(np.array([1, -100, 0], dtype=np.int64))
    axis.gradcheck(lambda x: ops.cross_entropy(x, targets), [t((3, 4))])


def test_cross_entropy_matches_manual():
    """CE must equal -mean(log_softmax[t]) exactly."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 7)).astype(np.float32)
    targets = np.array([1, 0, 6, 3, 2], dtype=np.int64)
    ce = ops.cross_entropy(Tensor(x), Tensor(targets)).item()
    shifted = x - x.max(-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    manual = float(-logp[np.arange(5), targets].mean())
    assert abs(ce - manual) < 1e-6
