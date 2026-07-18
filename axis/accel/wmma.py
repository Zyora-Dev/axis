"""Tensor-core (wmma) batched matmul — CUDA fast path.

locomp's `simdgroup_matrix_*` ops lower to CUDA `wmma::mma_sync` tensor-core
instructions (16x16x16 tiles, fp16 inputs / fp32 accumulate) — the same
hardware cuBLAS uses, but kept as our own portable kernel. One warp computes
one 16x16 output tile; the batch is folded into grid dim 1 so a whole
[B,M,K]@[B,K,N] is a single launch.

CUDA-ONLY on purpose: Metal maps simdgroup ops to 8x8 tiles (different size),
so this kernel's 16-tile indexing is only correct on CUDA. The scalar tiled
kernel stays the portable/reference path; this is the accelerated NVIDIA path.
Kept in its own module (locomp constexpr type isolation).
"""
import locomp

WMMA = 16  # CUDA tensor-core tile size (16x16x16)


@locomp.kernel
def wmma_matmul_b(A: locomp.Float16, B: locomp.Float16, C: locomp.Tensor,
                  M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
                  MTILES: locomp.constexpr):
    # grid = (N//16, BATCH * M//16), block = (32,)  (one warp per output tile).
    bcol = locomp.program_id(0)
    bblk = locomp.program_id(1)
    batch = bblk // MTILES
    trow = bblk % MTILES

    a_base = batch * M * K
    b_base = batch * K * N
    c_base = batch * M * N

    acc = locomp.simdgroup_matrix(0.0)
    nkt = K // 16
    for kt in range(nkt):
        a = locomp.simdgroup_matrix_load_device(
            A + (a_base + trow * 16 * K + kt * 16), K, role="a")
        b = locomp.simdgroup_matrix_load_device(
            B + (b_base + kt * 16 * N + bcol * 16), N, role="b")
        acc = locomp.simdgroup_mac(acc, a, b)
    locomp.simdgroup_matrix_store_device(acc, C + (c_base + trow * 16 * N + bcol * 16), N)
