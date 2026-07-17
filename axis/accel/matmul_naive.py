"""Isolated naive matmul (fallback for shapes below the tiled kernel's 16×16
tile). Kept alone — locomp leaks constexpr types across kernels sharing a
module.

2D grid (N, M) — one thread per output element, loop over K. A 3D grid
(batch, M, N) does NOT port to locomp's CUDA codegen (program_id(2) collapses
to 0), so the host loops the batch dimension instead.
"""
import locomp


@locomp.kernel
def nmm(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
        M: locomp.constexpr, K: locomp.constexpr, N: locomp.constexpr):
    col = locomp.program_id(0)  # n
    row = locomp.program_id(1)  # m
    acc = 0.0
    for k in range(K):
        a = locomp.load(A + row * K + k)
        b = locomp.load(B + k * N + col)
        acc = acc + a * b
    locomp.store(C + row * N + col, acc)
