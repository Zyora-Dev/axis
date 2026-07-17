"""Batched tiled matmul — one kernel launch for a whole [B, M, K] @ [B, K, N].

The per-batch host loop in accel.matmul was the dominant cost: it uploaded,
launched, and downloaded once *per batch element* (32x for attention). This
kernel folds the batch index into grid dimension 1, so the entire batch is a
single upload / launch / download.

Same proven 2D-grid + shared-memory idiom as tiled.py (which ports to CUDA);
the only addition is a batch offset derived from program_id(1). Kept in its own
module because locomp leaks constexpr param types across kernels in one file.
"""
import locomp

TILE = 16


@locomp.kernel
def tiled_matmul_b(A: locomp.Tensor, B: locomp.Tensor, C: locomp.Tensor,
                   M: locomp.constexpr, N: locomp.constexpr, K: locomp.constexpr,
                   NUM_TILES: locomp.constexpr, BLOCK: locomp.constexpr,
                   MTILES: locomp.constexpr):
    # grid = (N//TILE, BATCH * M//TILE), block = (TILE, TILE).
    # program_id(1) encodes both the batch and the row-tile within it.
    row = locomp.local_id(1)
    col = locomp.local_id(0)
    bcol = locomp.program_id(0)
    bblk = locomp.program_id(1)
    batch = bblk // MTILES
    brow = bblk % MTILES

    a_base = batch * M * K
    b_base = batch * K * N
    c_base = batch * M * N

    As = locomp.shared_memory(TILE * TILE)
    Bs = locomp.shared_memory(TILE * TILE)

    acc = 0.0
    for t in range(NUM_TILES):
        a_row = brow * BLOCK + row
        a_col = t * BLOCK + col
        a_val = locomp.load(A + (a_base + a_row * K + a_col))
        locomp.shared_store(As, row * BLOCK + col, a_val)

        b_row = t * BLOCK + row
        b_col = bcol * BLOCK + col
        b_val = locomp.load(B + (b_base + b_row * N + b_col))
        locomp.shared_store(Bs, row * BLOCK + col, b_val)

        locomp.barrier()
        for k in range(BLOCK):
            a_shared = locomp.shared_load(As, row * BLOCK + k)
            b_shared = locomp.shared_load(Bs, k * BLOCK + col)
            acc = acc + a_shared * b_shared
        locomp.barrier()

    out_idx = c_base + (brow * BLOCK + row) * N + (bcol * BLOCK + col)
    locomp.store(C + out_idx, acc)
