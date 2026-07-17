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


# ─── fused SiLU-multiply (SwiGLU inner): O = silu(G) * U ────────────────────


@locomp.kernel
def silu_mul_kernel(G: locomp.Tensor, U: locomp.Tensor, O: locomp.Tensor):
    i = locomp.program_id(0)
    g = locomp.load(G + i)
    locomp.store(O + i, (g / (1.0 + locomp.exp(-g))) * locomp.load(U + i))


# ─── fused causal attention ─────────────────────────────────────────────────
# One kernel = scores + scale + causal mask + stable softmax + weighted sum.
# Layout: Q,K,V,O are [BH, T, D] flattened; P (probs, saved for backward) is
# [BH, T, T]. Grid: (BH, T) — each program owns one query row and its own
# P[bh, t, :] slice, so global-memory scratch use is race-free.


@locomp.kernel
def fused_attn_kernel(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
                      O: locomp.Tensor, P: locomp.Tensor,
                      T: locomp.constexpr, D: locomp.constexpr,
                      SCALE: locomp.constexpr):
    bh = locomp.program_id(0)
    t = locomp.program_id(1)

    q_off = bh * T * D + t * D
    p_off = bh * T * T + t * T

    # Pass 0+1: scores into P (masked rows stay 0), track running max.
    mx = -1e30
    for j in range(T):
        if j <= t:
            s = 0.0
            for d in range(D):
                s = s + locomp.load(Q + q_off + d) * locomp.load(K + bh * T * D + j * D + d)
            s = s * SCALE
            locomp.store(P + p_off + j, s)
            mx = locomp.max(mx, s)
        else:
            locomp.store(P + p_off + j, 0.0)

    # Pass 2: exponentiate + sum.
    denom = 0.0
    for j in range(T):
        if j <= t:
            e = locomp.exp(locomp.load(P + p_off + j) - mx)
            locomp.store(P + p_off + j, e)
            denom = denom + e

    # Pass 3: normalize probs.
    for j in range(T):
        if j <= t:
            locomp.store(P + p_off + j, locomp.load(P + p_off + j) / denom)

    # Pass 4: O[t, :] = P[t, :] @ V.
    for d in range(D):
        acc = 0.0
        for j in range(T):
            if j <= t:
                acc = acc + locomp.load(P + p_off + j) * locomp.load(V + bh * T * D + j * D + d)
        locomp.store(O + q_off + d, acc)
