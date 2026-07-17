"""axis.accel — locomp GPU acceleration layer.

Contract:
- `axis.accel.enable()` turns GPU dispatch on; ops silently use locomp kernels
  where supported and fall back to NumPy everywhere else.
- GPU results MUST match the NumPy reference within float32 tolerance —
  enforced by tests/test_accel_parity.py.
- If locomp (or a GPU) is unavailable, everything still runs on NumPy: the
  reference engine is never optional.

Cross-vendor: the backend (cuda / rocm / metal) is auto-detected and threaded
explicitly through every tensor allocation AND kernel launch, because locomp's
own "auto" never selects ROCm. This is what lets Axis run on NVIDIA and AMD
from one codebase.
"""
from __future__ import annotations

import platform
import shutil

import numpy as np

_ENABLED = False
_AVAILABLE: bool | None = None
_BACKEND: str | None = None       # "cuda" | "rocm" | "metal"
_LAUNCHERS: dict = {}             # (func_id, backend) -> KernelLauncher


def detect_backend() -> str | None:
    """Pick the locomp backend for this machine. AMD (ROCm) must be chosen
    explicitly — locomp's 'auto' only resolves cuda/metal."""
    if platform.system() == "Darwin":
        return "metal"
    # AMD ROCm: hipcc / rocminfo present.
    if shutil.which("hipcc") or shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return "rocm"
    # NVIDIA CUDA: nvcc (to compile) — nvidia-smi alone isn't enough.
    if shutil.which("nvcc"):
        return "cuda"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return None


def backend() -> str | None:
    return _BACKEND


def _launcher(func, backend_name: str):
    """Return a locomp KernelLauncher for `func` bound to `backend_name`,
    cached. Rebuilds from the raw function so ROCm/CUDA get the right target
    (a launcher's backend is fixed at decoration time)."""
    key = (id(func), backend_name)
    launcher = _LAUNCHERS.get(key)
    if launcher is None:
        import locomp
        launcher = locomp.kernel(func, backend=backend_name)
        _LAUNCHERS[key] = launcher
    return launcher


def available() -> bool:
    """True if locomp is importable and the detected backend actually runs."""
    global _AVAILABLE, _BACKEND
    if _AVAILABLE is None:
        try:
            import locomp as lc
            from axis.accel import kernels
            _BACKEND = detect_backend()
            if _BACKEND is None:
                _AVAILABLE = False
                return _AVAILABLE
            # Probe: allocate + launch on the detected backend.
            x = lc.tensor(np.zeros(4, dtype=np.float32), backend=_BACKEND)
            o = lc.empty(4, backend=_BACKEND)
            _launcher(kernels.silu_kernel.func, _BACKEND)[(4,)](x, o)
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
    falls back to NumPy). Uses the shared-memory tiled kernel for larger
    shapes, the naive kernel for tiny ones."""
    if not _ENABLED:
        return None
    if a.ndim < 2 or b.ndim < 2 or a.dtype != np.float32 or b.dtype != np.float32:
        return None
    try:
        import locomp as lc
        from axis.accel.matmul_naive import nmm
        be = _BACKEND

        # Normalize to [B, M, K] @ [B, K, N]. When b is a shared 2D matrix
        # (the common Linear-layer case) collapse ALL of a's leading dims into
        # one big M: a single 2D matmul, weight uploaded once — no per-batch
        # broadcast/re-upload.
        N = b.shape[-1]
        out_shape = (*a.shape[:-1], N)
        if b.ndim == 2:
            K = a.shape[-1]
            if b.shape[0] != K:
                return None
            a3 = a.reshape(1, -1, K)
            b3 = b.reshape(1, K, N)
        else:
            a3 = a.reshape((-1, a.shape[-2], a.shape[-1]))
            b3 = b.reshape((-1, b.shape[-2], b.shape[-1]))
            if a3.shape[0] != b3.shape[0] or a3.shape[2] != b3.shape[1]:
                return None
        B, M, K = a3.shape
        N = b3.shape[2]

        if M >= 16 and N >= 16 and K >= 16:
            from axis.accel.batched_tiled import TILE, tiled_matmul_b
            launch = _launcher(tiled_matmul_b.func, be)
            # Pad M/N/K to a multiple of TILE (the tiled kernel assumes it).
            Mp = (M + TILE - 1) // TILE * TILE
            Np = (N + TILE - 1) // TILE * TILE
            Kp = (K + TILE - 1) // TILE * TILE
            nt = Kp // TILE
            mtiles = Mp // TILE
            tg = (TILE, TILE)
            # Pad + upload the WHOLE batch once, one launch, one download.
            if (Mp, Kp) != (M, K):
                apad = np.zeros((B, Mp, Kp), dtype=np.float32); apad[:, :M, :K] = a3
            else:
                apad = np.ascontiguousarray(a3)
            if (Kp, Np) != (K, N):
                bpad = np.zeros((B, Kp, Np), dtype=np.float32); bpad[:, :K, :N] = b3
            else:
                bpad = np.ascontiguousarray(b3)
            tta = lc.tensor(apad.reshape(-1), backend=be)
            ttb = lc.tensor(bpad.reshape(-1), backend=be)
            ttc = lc.empty(B * Mp * Np, backend=be)
            grid = (Np // TILE, B * mtiles)   # batch folded into grid dim 1
            launch[grid, tg](tta, ttb, ttc, Mp, Np, Kp, nt, TILE, mtiles)
            res = ttc.numpy().reshape(B, Mp, Np)[:, :M, :N]
        else:
            # Small shapes: 2D-grid naive kernel, host loops the batch (a 3D
            # grid does not port to locomp's CUDA codegen).
            launch = _launcher(nmm.func, be)
            res = np.empty((B, M, N), dtype=np.float32)
            grid = (N, M)  # (cols, rows)
            for bi in range(B):
                tta = lc.tensor(np.ascontiguousarray(a3[bi]).flatten(), backend=be)
                ttb = lc.tensor(np.ascontiguousarray(b3[bi]).flatten(), backend=be)
                ttc = lc.empty(M * N, backend=be)
                launch[grid](tta, ttb, ttc, M=M, K=K, N=N)
                res[bi] = ttc.numpy().reshape(M, N)
        return res.reshape(out_shape).astype(np.float32)
    except Exception:  # noqa: BLE001 — fall back to NumPy on any kernel issue
        return None


def softmax_lastdim(x: np.ndarray) -> np.ndarray | None:
    if not _ENABLED or x.dtype != np.float32:
        return None
    try:
        import locomp as lc
        from axis.accel.softmax import softmax_rows
        be = _BACKEND

        d = x.shape[-1]
        rows = x.reshape(-1, d)
        tx = lc.tensor(np.ascontiguousarray(rows), backend=be)
        to = lc.empty(rows.shape, backend=be)
        _launcher(softmax_rows.func, be)[(rows.shape[0],)](tx, to, D=d)
        return to.numpy().reshape(x.shape).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def silu(x: np.ndarray) -> np.ndarray | None:
    if not _ENABLED or x.dtype != np.float32:
        return None
    try:
        import locomp as lc
        from axis.accel.kernels import silu_kernel
        be = _BACKEND

        flat = np.ascontiguousarray(x.reshape(-1))
        tx = lc.tensor(flat, backend=be)
        to = lc.empty(flat.shape, backend=be)
        _launcher(silu_kernel.func, be)[(flat.size,)](tx, to)
        return to.numpy().reshape(x.shape).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def gelu(x: np.ndarray) -> np.ndarray | None:
    if not _ENABLED or x.dtype != np.float32:
        return None
    try:
        import locomp as lc
        from axis.accel.kernels import gelu_kernel
        be = _BACKEND

        flat = np.ascontiguousarray(x.reshape(-1))
        tx = lc.tensor(flat, backend=be)
        to = lc.empty(flat.shape, backend=be)
        _launcher(gelu_kernel.func, be)[(flat.size,)](tx, to)
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
        be = _BACKEND

        gf = np.ascontiguousarray(g.reshape(-1))
        uf = np.ascontiguousarray(u.reshape(-1))
        tg = lc.tensor(gf, backend=be)
        tu = lc.tensor(uf, backend=be)
        to = lc.empty(gf.shape, backend=be)
        _launcher(silu_mul_kernel.func, be)[(gf.size,)](tg, tu, to)
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
        from axis.accel.attention import fused_attn
        be = _BACKEND

        B, H, T, D = q.shape
        BH = B * H
        tq = lc.tensor(np.ascontiguousarray(q.reshape(BH, T, D)), backend=be)
        tk = lc.tensor(np.ascontiguousarray(k.reshape(BH, T, D)), backend=be)
        tv = lc.tensor(np.ascontiguousarray(v.reshape(BH, T, D)), backend=be)
        to = lc.empty((BH, T, D), backend=be)
        tp = lc.empty((BH, T, T), backend=be)
        _launcher(fused_attn.func, be)[(BH, T)](tq, tk, tv, to, tp, T=T, D=D, SCALE=float(scale))
        out = to.numpy().reshape(B, H, T, D).astype(np.float32)
        probs = tp.numpy().reshape(B, H, T, T).astype(np.float32)
        return out, probs
    except Exception:  # noqa: BLE001
        return None
