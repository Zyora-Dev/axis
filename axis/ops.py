"""Axis ops — differentiable primitives.

Every op:
  1. computes forward with NumPy,
  2. attaches a backward closure returning [(parent, parent_grad), ...],
  3. is covered by gradcheck tests against central finite differences.

Broadcasting is handled explicitly: `_unbroadcast` reduces an output gradient
back to the parent's original shape, so no silent shape bugs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from axis.tensor import Tensor, is_grad_enabled


def _make(data: np.ndarray, parents: Sequence[Tensor], op: str, backward) -> Tensor:
    req = is_grad_enabled() and any(p.requires_grad for p in parents)
    out = Tensor(data, requires_grad=req, _parents=parents if req else (), _op=op)
    if req:
        out._backward = backward
    return out


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` down to `shape` (reverse of numpy broadcasting)."""
    if grad.shape == shape:
        return grad
    # Sum leading extra dims.
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    # Sum broadcast (size-1) dims.
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


# ─── elementwise binary ─────────────────────────────────────────────────────


def add(a: Tensor, b: Tensor) -> Tensor:
    data = a.data + b.data
    def backward(g):
        return [(a, _unbroadcast(g, a.shape)), (b, _unbroadcast(g, b.shape))]
    return _make(data, (a, b), "add", backward)


def sub(a: Tensor, b: Tensor) -> Tensor:
    data = a.data - b.data
    def backward(g):
        return [(a, _unbroadcast(g, a.shape)), (b, _unbroadcast(-g, b.shape))]
    return _make(data, (a, b), "sub", backward)


def mul(a: Tensor, b: Tensor) -> Tensor:
    data = a.data * b.data
    def backward(g):
        return [
            (a, _unbroadcast(g * b.data, a.shape)),
            (b, _unbroadcast(g * a.data, b.shape)),
        ]
    return _make(data, (a, b), "mul", backward)


def div(a: Tensor, b: Tensor) -> Tensor:
    data = a.data / b.data
    def backward(g):
        return [
            (a, _unbroadcast(g / b.data, a.shape)),
            (b, _unbroadcast(-g * a.data / (b.data * b.data), b.shape)),
        ]
    return _make(data, (a, b), "div", backward)


def pow(a: Tensor, p: float) -> Tensor:  # noqa: A001
    data = a.data ** p
    def backward(g):
        return [(a, g * p * (a.data ** (p - 1)))]
    return _make(data, (a,), f"pow{p}", backward)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    data = np.maximum(a.data, b.data)
    def backward(g):
        mask = (a.data >= b.data).astype(g.dtype)
        return [
            (a, _unbroadcast(g * mask, a.shape)),
            (b, _unbroadcast(g * (1.0 - mask), b.shape)),
        ]
    return _make(data, (a, b), "maximum", backward)


def where(cond: np.ndarray, a: Tensor, b: Tensor) -> Tensor:
    c = np.asarray(cond)
    data = np.where(c, a.data, b.data)
    def backward(g):
        return [
            (a, _unbroadcast(g * c, a.shape)),
            (b, _unbroadcast(g * (~c if c.dtype == np.bool_ else 1 - c), b.shape)),
        ]
    return _make(data, (a, b), "where", backward)


# ─── elementwise unary ──────────────────────────────────────────────────────


def exp(a: Tensor) -> Tensor:
    data = np.exp(a.data)
    def backward(g):
        return [(a, g * data)]
    return _make(data, (a,), "exp", backward)


def log(a: Tensor) -> Tensor:
    data = np.log(a.data)
    def backward(g):
        return [(a, g / a.data)]
    return _make(data, (a,), "log", backward)


def sqrt(a: Tensor) -> Tensor:
    data = np.sqrt(a.data)
    def backward(g):
        return [(a, g * 0.5 / data)]
    return _make(data, (a,), "sqrt", backward)


def tanh(a: Tensor) -> Tensor:
    data = np.tanh(a.data)
    def backward(g):
        return [(a, g * (1.0 - data * data))]
    return _make(data, (a,), "tanh", backward)


def sigmoid(a: Tensor) -> Tensor:
    data = 1.0 / (1.0 + np.exp(-a.data))
    def backward(g):
        return [(a, g * data * (1.0 - data))]
    return _make(data, (a,), "sigmoid", backward)


def relu(a: Tensor) -> Tensor:
    data = np.maximum(a.data, 0.0)
    def backward(g):
        return [(a, g * (a.data > 0))]
    return _make(data, (a,), "relu", backward)


def gelu(a: Tensor) -> Tensor:
    """GELU (tanh approximation — the form used by GPT-2/Gemma)."""
    c = np.float32(0.7978845608028654)  # sqrt(2/pi)
    k = np.float32(0.044715)
    x = a.data
    inner = c * (x + k * x**3)
    t = np.tanh(inner)
    data = 0.5 * x * (1.0 + t)
    def backward(g):
        # d/dx [0.5x(1+t)] = 0.5(1+t) + 0.5x * (1-t^2) * c(1+3k x^2)
        dt = (1.0 - t * t) * c * (1.0 + 3.0 * k * x * x)
        return [(a, g * (0.5 * (1.0 + t) + 0.5 * x * dt))]
    return _make(data, (a,), "gelu", backward)


def silu(a: Tensor) -> Tensor:
    """SiLU / swish: x * sigmoid(x). Used by SwiGLU (Llama-family MLPs)."""
    sig = 1.0 / (1.0 + np.exp(-a.data))
    data = a.data * sig
    def backward(g):
        return [(a, g * (sig * (1.0 + a.data * (1.0 - sig))))]
    return _make(data, (a,), "silu", backward)


# ─── matmul ─────────────────────────────────────────────────────────────────


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Batched matmul with full broadcast support (numpy semantics)."""
    data = a.data @ b.data
    def backward(g):
        # dA = g @ B^T ; dB = A^T @ g  (with batch-dim reduction)
        b_t = np.swapaxes(b.data, -1, -2)
        a_t = np.swapaxes(a.data, -1, -2)
        ga = g @ b_t
        gb = a_t @ g
        return [(a, _unbroadcast(ga, a.shape)), (b, _unbroadcast(gb, b.shape))]
    return _make(data, (a, b), "matmul", backward)


# ─── reductions ─────────────────────────────────────────────────────────────


def sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:  # noqa: A001
    data = a.data.sum(axis=axis, keepdims=keepdims)
    def backward(g):
        if axis is None:
            return [(a, np.broadcast_to(g, a.shape).copy())]
        gg = g if keepdims else np.expand_dims(g, axis)
        return [(a, np.broadcast_to(gg, a.shape).copy())]
    return _make(data, (a,), "sum", backward)


def mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    n = a.data.size if axis is None else a.data.shape[axis]
    data = a.data.mean(axis=axis, keepdims=keepdims)
    def backward(g):
        if axis is None:
            return [(a, np.broadcast_to(g / n, a.shape).copy())]
        gg = g if keepdims else np.expand_dims(g, axis)
        return [(a, np.broadcast_to(gg / n, a.shape).copy())]
    return _make(data, (a,), "mean", backward)


def max(a: Tensor, axis: int, keepdims: bool = False) -> Tensor:  # noqa: A001
    data = a.data.max(axis=axis, keepdims=keepdims)
    def backward(g):
        expanded = data if keepdims else np.expand_dims(data, axis)
        mask = (a.data == expanded).astype(g.dtype)
        # Split gradient equally among ties (deterministic).
        mask /= mask.sum(axis=axis, keepdims=True)
        gg = g if keepdims else np.expand_dims(g, axis)
        return [(a, mask * gg)]
    return _make(data, (a,), "max", backward)


# ─── shape ──────────────────────────────────────────────────────────────────


def reshape(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    data = a.data.reshape(shape)
    def backward(g):
        return [(a, g.reshape(a.shape))]
    return _make(data, (a,), "reshape", backward)


def transpose(a: Tensor, d0: int = -2, d1: int = -1) -> Tensor:
    data = np.swapaxes(a.data, d0, d1)
    def backward(g):
        return [(a, np.swapaxes(g, d0, d1))]
    return _make(data, (a,), "transpose", backward)


def getitem(a: Tensor, idx) -> Tensor:
    data = a.data[idx]
    def backward(g):
        full = np.zeros_like(a.data, dtype=g.dtype)
        np.add.at(full, idx, g)
        return [(a, full)]
    return _make(data, (a,), "getitem", backward)


def cat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    data = np.concatenate([t.data for t in tensors], axis=axis)
    sizes = [t.data.shape[axis] for t in tensors]
    def backward(g):
        splits = np.split(g, np.cumsum(sizes)[:-1], axis=axis)
        return list(zip(tensors, splits))
    return _make(data, tuple(tensors), "cat", backward)


# ─── softmax family ─────────────────────────────────────────────────────────


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    shifted = a.data - a.data.max(axis=axis, keepdims=True)  # stability
    e = np.exp(shifted)
    data = e / e.sum(axis=axis, keepdims=True)
    def backward(g):
        # dx = s * (g - sum(g * s))
        dot = (g * data).sum(axis=axis, keepdims=True)
        return [(a, data * (g - dot))]
    return _make(data, (a,), "softmax", backward)


def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    data = shifted - lse
    def backward(g):
        s = np.exp(data)
        return [(a, g - s * g.sum(axis=axis, keepdims=True))]
    return _make(data, (a,), "log_softmax", backward)


# ─── embedding + cross-entropy (the two index-based ops) ────────────────────


def embedding(weight: Tensor, indices: Tensor) -> Tensor:
    """weight[V, D] gathered by integer indices[...]. Backward = scatter-add."""
    idx = indices.data.astype(np.int64)
    data = weight.data[idx]
    def backward(g):
        gw = np.zeros_like(weight.data)
        np.add.at(gw, idx, g)
        return [(weight, gw)]
    return _make(data, (weight,), "embedding", backward)


def cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """Mean token-level cross-entropy.

    logits: [N, V] float. targets: [N] int64. Fused log-softmax + NLL for
    numerical stability (never materializes softmax probabilities in fp32 sums).
    """
    t = targets.data.astype(np.int64).reshape(-1)
    x = logits.data.reshape(-1, logits.shape[-1])
    valid = t != ignore_index
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("cross_entropy: no valid targets")

    shifted = x - x.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    logp = shifted - lse  # [N, V]
    safe_t = np.where(valid, t, 0)
    nll = -logp[np.arange(x.shape[0]), safe_t]
    nll = np.where(valid, nll, 0.0)
    data = np.float32(nll.sum() / n_valid)

    def backward(g):
        # d/dx = (softmax(x) - onehot(t)) / n_valid, zeroed on ignored rows
        p = np.exp(logp)
        p[np.arange(x.shape[0]), safe_t] -= 1.0
        p *= (valid[:, None] / n_valid)
        return [(logits, (g * p).reshape(logits.shape).astype(np.float32))]

    return _make(data, (logits,), "cross_entropy", backward)
