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

import numpy as _np

from axis.backend import xp as np, array_module, is_gpu_array, scatter_add
from axis.tensor import Tensor, is_grad_enabled
from axis import accel


def _mm(a, b):
    """Matmul. On GPU arrays this is cuBLAS via `a @ b`; on CPU it tries the
    locomp kernel then falls back to NumPy BLAS."""
    if is_gpu_array(a) or is_gpu_array(b):
        return a @ b
    out = accel.matmul(a, b)
    return out if out is not None else a @ b


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
    gpu = accel.gelu(x)
    data = gpu if gpu is not None else 0.5 * x * (1.0 + t)
    def backward(g):
        # d/dx [0.5x(1+t)] = 0.5(1+t) + 0.5x * (1-t^2) * c(1+3k x^2)
        dt = (1.0 - t * t) * c * (1.0 + 3.0 * k * x * x)
        return [(a, g * (0.5 * (1.0 + t) + 0.5 * x * dt))]
    return _make(data, (a,), "gelu", backward)


def silu(a: Tensor) -> Tensor:
    """SiLU / swish: x * sigmoid(x). Used by SwiGLU (Llama-family MLPs)."""
    sig = 1.0 / (1.0 + np.exp(-a.data))
    gpu = accel.silu(a.data)
    data = gpu if gpu is not None else a.data * sig
    def backward(g):
        return [(a, g * (sig * (1.0 + a.data * (1.0 - sig))))]
    return _make(data, (a,), "silu", backward)


# ─── matmul ─────────────────────────────────────────────────────────────────


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Batched matmul with full broadcast support (numpy semantics).
    Forward AND backward matmuls route through the GPU when enabled. When
    residency applies, the result is kept on the GPU (out._dev) and shared
    inputs are uploaded once."""
    dev = None
    res = accel.matmul_resident(a, b) if accel.is_enabled() else None
    if res is not None:
        data, dev = res
    else:
        data = _mm(a.data, b.data)
    def backward(g):
        # dA = g @ B^T ; dB = A^T @ g  (with batch-dim reduction)
        b_t = np.swapaxes(b.data, -1, -2)
        a_t = np.swapaxes(a.data, -1, -2)
        ga = _mm(g, np.ascontiguousarray(b_t))
        gb = _mm(np.ascontiguousarray(a_t), g)
        return [(a, _unbroadcast(ga, a.shape)), (b, _unbroadcast(gb, b.shape))]
    out = _make(data, (a, b), "matmul", backward)
    out._dev = dev
    return out


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
        full = array_module(a.data).zeros_like(a.data, dtype=g.dtype)
        scatter_add(full, idx, g)
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
    if axis in (-1, a.data.ndim - 1):
        gpu = accel.softmax_lastdim(a.data)
    else:
        gpu = None
    if gpu is not None:
        data = gpu
    else:
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
    xp = array_module(weight.data)
    idx = xp.asarray(indices.data).astype(_np.int64)  # index on the same device
    data = weight.data[idx]
    def backward(g):
        gw = xp.zeros_like(weight.data)
        scatter_add(gw, idx, g)
        return [(weight, gw)]
    return _make(data, (weight,), "embedding", backward)


def cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """Mean token-level cross-entropy.

    logits: [N, V] float. targets: [N] int64. Fused log-softmax + NLL for
    numerical stability (never materializes softmax probabilities in fp32 sums).
    """
    xp = array_module(logits.data)
    t = xp.asarray(targets.data).astype(_np.int64).reshape(-1)
    x = logits.data.reshape(-1, logits.shape[-1])
    if x.dtype == _np.float16:      # CE always in fp32 (AMP practice)
        x = x.astype(_np.float32)
    valid = t != ignore_index
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("cross_entropy: no valid targets")

    shifted = x - x.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    logp = shifted - lse  # [N, V]
    rows = xp.arange(x.shape[0])
    safe_t = xp.where(valid, t, 0)
    nll = -logp[rows, safe_t]
    nll = xp.where(valid, nll, 0.0)
    data = _np.float32(float(nll.sum()) / n_valid)

    def backward(g):
        # d/dx = (softmax(x) - onehot(t)) / n_valid, zeroed on ignored rows
        p = xp.exp(logp)
        p[rows, safe_t] -= 1.0
        p *= (valid[:, None] / n_valid)
        return [(logits, (g * p).reshape(logits.shape).astype(logits.data.dtype))]

    return _make(data, (logits,), "cross_entropy", backward)


# ─── fused ops (Phase 2.5 — one tape node, one GPU round trip) ─────────────


def silu_mul(g: Tensor, u: Tensor) -> Tensor:
    """Fused silu(g) * u — the SwiGLU inner product. Computed in the storage
    dtype: fp16 sigmoid is safe (exp overflow/underflow saturate to the exact
    limits 0/1), so no fp32 copy is needed."""
    sig = 1.0 / (1.0 + np.exp(-g.data))
    gpu = accel.silu_mul(g.data, u.data)
    data = gpu if gpu is not None else (g.data * sig) * u.data
    def backward(grad):
        silu_g = g.data * sig
        d_silu = sig * (1.0 + g.data * (1.0 - sig))
        return [
            (g, grad * u.data * d_silu),
            (u, grad * silu_g),
        ]
    return _make(data, (g, u), "silu_mul", backward)


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Fused RMSNorm: x / sqrt(mean(x^2, -1) + eps) * weight.

    ONE tape node with an exact analytic backward. Precision-critical
    REDUCTIONS accumulate in fp32 (dtype= on mean/sum — no fp32 copy of x is
    materialized), storage stays in the input dtype (fp16 under AMP).
    """
    dt = x.data.dtype
    xd = x.data
    wd = weight.data
    D = xd.shape[-1]
    ms = (xd * xd).mean(axis=-1, keepdims=True, dtype=_np.float32)
    inv = (ms + _np.float32(eps)) ** -0.5          # [..., 1] fp32, tiny
    inv_dt = inv.astype(dt)
    data = xd * inv_dt * wd

    def backward(g):
        gw_x = (g * wd * xd).sum(axis=-1, keepdims=True, dtype=_np.float32)
        dx = g * wd * inv_dt - xd * ((inv ** 3) * (gw_x / D)).astype(dt)
        dw = _unbroadcast((g * xd * inv_dt).astype(_np.float32), weight.shape)
        return [(x, dx.astype(dt)), (weight, dw.astype(wd.dtype))]

    return _make(data, (x, weight), "rmsnorm", backward)


def rope(x: Tensor, cos, sin) -> Tensor:
    """Fused rotary position embedding on [B, H, T, D] (half-split rotate).

    ONE tape node replacing ~10 ops (2 getitem, 4 mul, sub, add, cat). The
    rotation is linear, so backward is the exact inverse rotation of the grad.
    cos/sin: [T, D/2] arrays (host or device — moved to x's device).
    """
    xp = array_module(x.data)
    dt = x.data.dtype
    T = x.shape[2]
    d_half = x.shape[-1] // 2
    c = xp.asarray(cos[None, None, :T, :]).astype(dt)   # rotation coeffs are
    s = xp.asarray(sin[None, None, :T, :]).astype(dt)   # bounded — fp16-safe
    x1 = x.data[..., :d_half]
    x2 = x.data[..., d_half:]
    data = xp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)

    def backward(g):
        g1 = g[..., :d_half]
        g2 = g[..., d_half:]
        dx = xp.concatenate([g1 * c + g2 * s, -g1 * s + g2 * c], axis=-1)
        return [(x, dx.astype(dt))]

    return _make(data, (x,), "rope", backward)


def fused_causal_attention(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Causal attention in one op: softmax(mask(q@k^T * scale)) @ v.

    q, k, v: [B, H, T, D]. Flash-attention-style memory: the O(T^2)
    probability matrix is NOT saved for backward — it is recomputed from q and
    k (bit-identical math, so gradients are unchanged). This removes the
    dominant [B, H, T, T] per-layer activation from peak memory, at the cost of
    one extra matmul + softmax in backward.
    """
    B, H, T, D = q.shape
    dt = q.data.dtype

    def _probs(qd, kd):
        xp = array_module(qd)
        # matmul in storage dtype (fp16 tensor cores under AMP); softmax is
        # shift-stabilized in the storage dtype with fp32-ACCUMULATED sum —
        # post-shift values are ≤ 0 so exp ≤ 1 (fp16-safe), no fp32 copy of
        # the [B,H,T,T] scores is materialized.
        scores = _mm(qd, np.ascontiguousarray(np.swapaxes(kd, -1, -2))) * dt.type(scale)
        mask = xp.triu(xp.full((T, T), -1e4 if dt == _np.float16 else -1e30, dtype=dt), k=1)
        scores = scores + mask[None, None]
        shifted = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        denom = e.sum(axis=-1, keepdims=True, dtype=_np.float32)
        return (e / denom.astype(dt)) if dt == _np.float16 else (e / denom).astype(dt)

    fused = accel.fused_causal_attention(q.data, k.data, v.data, scale)
    if fused is not None:
        data, _ = fused          # discard probs — recomputed in backward
    else:
        data = _mm(_probs(q.data, k.data), v.data)

    def backward(grad):
        # Recompute P from q, k (parents — still alive during this backward).
        probs = _probs(q.data, k.data)
        # dV = P^T @ dO
        p_t = np.ascontiguousarray(np.swapaxes(probs, -1, -2))
        dv = _mm(p_t, grad)
        # dP = dO @ V^T ; softmax backward: dS = P * (dP - sum(dP*P))
        dp = _mm(grad, np.ascontiguousarray(np.swapaxes(v.data, -1, -2)))
        ds = (probs * (dp - (dp * probs).sum(axis=-1, keepdims=True))).astype(dt)
        # dQ = dS @ K * scale ; dK = dS^T @ Q * scale
        dq = _mm(ds, k.data) * scale
        dk = _mm(np.ascontiguousarray(np.swapaxes(ds, -1, -2)), q.data) * scale
        return [(q, dq.astype(dt)), (k, dk.astype(dt)), (v, dv.astype(dt))]

    return _make(data, (q, k, v), "fused_causal_attention", backward)
