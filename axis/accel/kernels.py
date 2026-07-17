"""Axis GPU kernels — written in locomp (Metal / CUDA / ROCm / RISC-V).

Phase 2 design: correctness first. Kernels are intentionally simple, flat-
indexed and single-purpose; every result is parity-tested against the NumPy
reference engine. Performance tuning (tiling, threadgroups, fusion) comes
after parity is locked in.
"""
from __future__ import annotations

import locomp

# ─── batched matmul: A[B,M,K] @ B[B,K,N] → O[B,M,N] ─────────────────────────


@locomp.kernel
def bmm_kernel(A: locomp.Tensor, B: locomp.Tensor, O: locomp.Tensor,
               M: locomp.constexpr, K: locomp.constexpr, N: locomp.constexpr):
    b = locomp.program_id(0)
    m = locomp.program_id(1)
    n = locomp.program_id(2)
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + b * M * K + m * K + k) * locomp.load(B + b * K * N + k * N + n)
    locomp.store(O + b * M * N + m * N + n, acc)


# ─── row-wise numerically-stable softmax: X[R,D] → O[R,D] ───────────────────


@locomp.kernel
def softmax_rows_kernel(X: locomp.Tensor, O: locomp.Tensor, D: locomp.constexpr):
    row = locomp.program_id(0)
    mx = locomp.load(X + row * D)
    for i in range(1, D):
        v = locomp.load(X + row * D + i)
        mx = locomp.max(mx, v)
    s = 0.0
    for i in range(D):
        s = s + locomp.exp(locomp.load(X + row * D + i) - mx)
    for i in range(D):
        e = locomp.exp(locomp.load(X + row * D + i) - mx)
        locomp.store(O + row * D + i, e / s)


# ─── elementwise ────────────────────────────────────────────────────────────


@locomp.kernel
def silu_kernel(X: locomp.Tensor, O: locomp.Tensor):
    i = locomp.program_id(0)
    v = locomp.load(X + i)
    locomp.store(O + i, v / (1.0 + locomp.exp(-v)))


@locomp.kernel
def gelu_kernel(X: locomp.Tensor, O: locomp.Tensor):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = locomp.tanh(inner)
    locomp.store(O + i, 0.5 * x * (1.0 + t))
