"""Isolated fused causal attention kernel. Kept alone — locomp leaks constexpr
types across kernels sharing a module.

One kernel = scores + scale + causal mask + stable softmax + weighted sum.
Layout: Q,K,V,O are [BH, T, D] flattened; P (probs, saved for backward) is
[BH, T, T]. Grid (BH, T): each program owns one query row and its own
P[bh, t, :] slice, so global scratch is race-free.
"""
import locomp


@locomp.kernel
def fused_attn(Q: locomp.Tensor, K: locomp.Tensor, V: locomp.Tensor,
               O: locomp.Tensor, P: locomp.Tensor,
               T: locomp.constexpr, D: locomp.constexpr, SCALE: locomp.constexpr):
    bh = locomp.program_id(0)
    t = locomp.program_id(1)

    q_off = bh * T * D + t * D
    p_off = bh * T * T + t * T

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

    denom = 0.0
    for j in range(T):
        if j <= t:
            e = locomp.exp(locomp.load(P + p_off + j) - mx)
            locomp.store(P + p_off + j, e)
            denom = denom + e

    for j in range(T):
        if j <= t:
            locomp.store(P + p_off + j, locomp.load(P + p_off + j) / denom)

    for d in range(D):
        acc = 0.0
        for j in range(T):
            if j <= t:
                acc = acc + locomp.load(P + p_off + j) * locomp.load(V + bh * T * D + j * D + d)
        locomp.store(O + q_off + d, acc)
