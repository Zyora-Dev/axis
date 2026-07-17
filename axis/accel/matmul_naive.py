"""Isolated naive batched matmul (fallback for tiny shapes). Kept alone —
locomp leaks constexpr types across kernels sharing a module."""
import locomp


@locomp.kernel
def bmm(A: locomp.Tensor, B: locomp.Tensor, O: locomp.Tensor,
        M: locomp.constexpr, K: locomp.constexpr, N: locomp.constexpr):
    b = locomp.program_id(0)
    m = locomp.program_id(1)
    n = locomp.program_id(2)
    acc = 0.0
    for k in range(K):
        acc = acc + locomp.load(A + b * M * K + m * K + k) * locomp.load(B + b * K * N + k * N + n)
    locomp.store(O + b * M * N + m * N + n, acc)
