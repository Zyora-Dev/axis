"""Isolated tiled matmul kernel.

Kept in its own module: locomp's constexpr type inference can leak across
kernels defined in the same file, so the shared-memory tiled matmul (which
relies heavily on constexpr dims) lives alone. Verbatim idiom from locomp's
proven example 07 — NUM_TILES + BLOCK explicit constexpr, dims padded to a
multiple of TILE by the host, single 2D matmul (host loops the batch).
"""
import locomp

TILE = 16


@locomp.kernel
def tiled_matmul(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                 M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
                 NUM_TILES: locomp.constexpr, BLOCK: locomp.constexpr):
    row = locomp.local_id(1)
    col = locomp.local_id(0)
    brow = locomp.program_id(1)
    bcol = locomp.program_id(0)

    As = locomp.shared_memory(TILE * TILE)
    Bs = locomp.shared_memory(TILE * TILE)

    acc = 0.0
    for t in range(NUM_TILES):
        a_row = brow * BLOCK + row
        a_col = t * BLOCK + col
        a_val = locomp.load(A + (a_row * K + a_col))
        locomp.shared_store(As, row * BLOCK + col, a_val)

        b_row = t * BLOCK + row
        b_col = bcol * BLOCK + col
        b_val = locomp.load(B + (b_row * N + b_col))
        locomp.shared_store(Bs, row * BLOCK + col, b_val)

        locomp.barrier()
        for k in range(BLOCK):
            a_shared = locomp.shared_load(As, row * BLOCK + k)
            b_shared = locomp.shared_load(Bs, k * BLOCK + col)
            acc = acc + a_shared * b_shared
        locomp.barrier()

    out_idx = (brow * BLOCK + row) * N + (bcol * BLOCK + col)
    locomp.store(C + out_idx, acc)
