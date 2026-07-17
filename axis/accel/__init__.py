"""axis.accel — locomp GPU acceleration layer.

Contract:
- `axis.accel.enable()` turns GPU dispatch on; ops silently use locomp kernels
  where supported and fall back to NumPy everywhere else.
- GPU results MUST match the NumPy reference within float32 tolerance —
  enforced by tests/test_accel_parity.py.
- If locomp (or a GPU) is unavailable, everything still runs on NumPy: the
  reference engine is never optional.
"""
from __future__ import annotations

import numpy as np

_ENABLED = False
_AVAILABLE: bool | None = None


def available() -> bool:
    """True if locomp is importable and a backend responds."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import locomp  # noqa: F401
            from axis.accel import kernels  # noqa: F401
            # Probe with a trivial launch.
            import locomp as lc
            x = lc.tensor(np.zeros(4, dtype=np.float32))
            o = lc.empty(4)
            kernels.silu_kernel[(4,)](x, o)
            _AVAILABLE = bool(np.allclose(o.numpy(), 0.0))
        except Exception:  # noqa: BLE001 — any failure = no GPU, never crash
            _AVAILABLE = False
    return _AVAILABLE


def enable() -> bool:
    """Enable GPU dispatch. Returns True if actually available."""
    global _ENABLED
    _ENABLED = available()
    return _ENABLED


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


# ─── array-level accelerated functions (NumPy in, NumPy out) ────────────────
# Phase 2 keeps autograd orchestration on CPU; these accelerate the heavy
# forward math. Device residency (keeping tensors on-GPU across ops) is the
# Phase 2.5 optimization once parity is proven.


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Batched matmul via locomp. Returns None if shapes unsupported (caller
    falls back to NumPy)."""
    if not _ENABLED:
        return None
    if a.ndim < 2 or b.ndim < 2 or a.dtype != np.float32 or b.dtype != np.float32:
        return None
    try:
        import locomp as lc
        from axis.accel.kernels import bmm_kernel

        # Normalize to [B, M, K] @ [B, K, N] via numpy broadcasting rules.
        a3 = a.reshape((-1, a.shape[-2], a.shape[-1])) if a.ndim > 2 else a[None]
        if b.ndim == 2:
            b3 = np.broadcast_to(b, (a3.shape[0], *b.shape))
        else:
            b3 = b.reshape((-1, b.shape[-2], b.shape[-1]))
        if a3.shape[0] != b3.shape[0] or a3.shape[2] != b3.shape[1]:
            return None
        B, M, K = a3.shape
        N = b3.shape[2]

        ta = lc.tensor(np.ascontiguousarray(a3))
        tb = lc.tensor(np.ascontiguousarray(b3))
        to = lc.empty((B, M, N))
        bmm_kernel[(B, M, N)](ta, tb, to, M=M, K=K, N=N)
        out = to.numpy().reshape((*a.shape[:-1], N) if a.ndim > 2 else (M, N))
        return out.astype(np.float32)
    except Exception:  # noqa: BLE001 — fall back to NumPy on any kernel issue
        return None


def softmax_lastdim(x: np.ndarray) -> np.ndarray | None:
    if not _ENABLED or x.dtype != np.float32:
        return None
    try:
        import locomp as lc
        from axis.accel.kernels import softmax_rows_kernel

        d = x.shape[-1]
        rows = x.reshape(-1, d)
        tx = lc.tensor(np.ascontiguousarray(rows))
        to = lc.empty(rows.shape)
        softmax_rows_kernel[(rows.shape[0],)](tx, to, D=d)
        return to.numpy().reshape(x.shape).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def silu(x: np.ndarray) -> np.ndarray | None:
    if not _ENABLED or x.dtype != np.float32:
        return None
    try:
        import locomp as lc
        from axis.accel.kernels import silu_kernel

        flat = np.ascontiguousarray(x.reshape(-1))
        tx = lc.tensor(flat)
        to = lc.empty(flat.shape)
        silu_kernel[(flat.size,)](tx, to)
        return to.numpy().reshape(x.shape).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def gelu(x: np.ndarray) -> np.ndarray | None:
    if not _ENABLED or x.dtype != np.float32:
        return None
    try:
        import locomp as lc
        from axis.accel.kernels import gelu_kernel

        flat = np.ascontiguousarray(x.reshape(-1))
        tx = lc.tensor(flat)
        to = lc.empty(flat.shape)
        gelu_kernel[(flat.size,)](tx, to)
        return to.numpy().reshape(x.shape).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def silu_mul(g: np.ndarray, u: np.ndarray) -> np.ndarray | None:
    """Fused silu(g) * u — the SwiGLU inner product, one round trip."""
    if not _ENABLED or g.dtype != np.float32 or g.shape != u.shape:
        return None
    try:
        import locomp as lc
        from axis.accel.kernels import silu_mul_kernel

        gf = np.ascontiguousarray(g.reshape(-1))
        uf = np.ascontiguousarray(u.reshape(-1))
        tg = lc.tensor(gf)
        tu = lc.tensor(uf)
        to = lc.empty(gf.shape)
        silu_mul_kernel[(gf.size,)](tg, tu, to)
        return to.numpy().reshape(g.shape).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def fused_causal_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fused causal attention forward.

    q, k, v: [B, H, T, D] float32. Returns (out [B,H,T,D], probs [B,H,T,T])
    in ONE kernel launch — versus 3 GPU round trips + CPU softmax composed.
    Probs are returned because the exact backward needs them.
    """
    if not _ENABLED or q.dtype != np.float32:
        return None
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        return None
    try:
        import locomp as lc
        from axis.accel.kernels import fused_attn_kernel

        B, H, T, D = q.shape
        BH = B * H
        tq = lc.tensor(np.ascontiguousarray(q.reshape(BH, T, D)))
        tk = lc.tensor(np.ascontiguousarray(k.reshape(BH, T, D)))
        tv = lc.tensor(np.ascontiguousarray(v.reshape(BH, T, D)))
        to = lc.empty((BH, T, D))
        tp = lc.empty((BH, T, T))
        fused_attn_kernel[(BH, T)](tq, tk, tv, to, tp, T=T, D=D, SCALE=float(scale))
        out = to.numpy().reshape(B, H, T, D).astype(np.float32)
        probs = tp.numpy().reshape(B, H, T, T).astype(np.float32)
        return out, probs
    except Exception:  # noqa: BLE001
        return None
