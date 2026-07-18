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
_TENSOR_CORES = False             # CUDA wmma fp16 path — OFF by default.
# Validated on A100: the wmma kernel is correct (train loss matches fp32 to
# 1e-4) and fast on its own (~0.15ms vs ~1.5ms scalar for 512^3). BUT end to
# end it's SLOWER, because per-op CPU<->GPU transfer is ~10x the kernel time
# and fp16 casting adds host work on top. Tensor cores only pay off once
# tensors stay resident on the GPU across ops (device residency). Kept here,
# opt-in via use_tensor_cores(True), as groundwork for that.


def use_tensor_cores(flag: bool) -> None:
    """Toggle the CUDA tensor-core (fp16) matmul fast path. Off by default:
    at present the per-op host round-trip dominates, so fp16 casting is a net
    loss end-to-end. Becomes a win with device residency."""
    global _TENSOR_CORES
    _TENSOR_CORES = bool(flag)


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


# ─── device residency ───────────────────────────────────────────────────────
# Upload (host->device) is ~10x the kernel time; download is cheap. We cache a
# flat device buffer on a Tensor so a value already on the GPU (e.g. the shared
# activation feeding q/k/v or gate/up) is uploaded ONCE, not per matmul.
# `.data` stays valid (we still download results), so backward + correctness
# are unchanged. Validated by the accel parity tests.

_RESIDENT = True                 # enable the residency cache
_up_count = 0                    # instrumentation: uploads performed
_hit_count = 0                   # instrumentation: cache hits (uploads avoided)


def residency_stats():
    return {"uploads": _up_count, "hits": _hit_count}


def reset_residency_stats():
    global _up_count, _hit_count
    _up_count = _hit_count = 0


def to_device(t):
    """Cached flat float32 device buffer for `t`. Uploads (and caches on
    `t._dev`) only on a miss. Returns None if residency isn't possible."""
    global _up_count, _hit_count
    if not _ENABLED or _BACKEND is None or not _RESIDENT:
        return None
    dev = getattr(t, "_dev", None)
    if dev is not None:
        _hit_count += 1
        return dev
    if t.data.dtype != np.float32:
        return None
    try:
        import locomp as lc
        arr = np.ascontiguousarray(t.data, dtype=np.float32).reshape(-1)
        dev = lc.tensor(arr, backend=_BACKEND)
        t._dev = dev
        _up_count += 1
        return dev
    except Exception:  # noqa: BLE001
        return None


def invalidate(t) -> None:
    """Drop the cached device buffer (call when `t.data` is mutated in place)."""
    t._dev = None


# ─── array-level accelerated functions (NumPy in, NumPy out) ────────────────
# Phase 2 keeps autograd orchestration on CPU; these accelerate the heavy
# forward math. Device residency (keeping tensors on-GPU across ops) is the
# Phase 2.5 optimization once parity is proven.


def _mm_wmma(a3, b3, B, M, N, K, be):
    """CUDA tensor-core batched matmul. Pads M/N/K to 16, casts inputs to
    fp16, accumulates in fp32, one launch for the whole batch. Returns
    [B, M, N] float32, or None on any failure (caller falls back to scalar)."""
    try:
        import locomp as lc
        from axis.accel.wmma import WMMA, wmma_matmul_b

        Mp = (M + WMMA - 1) // WMMA * WMMA
        Np = (N + WMMA - 1) // WMMA * WMMA
        Kp = (K + WMMA - 1) // WMMA * WMMA
        mtiles = Mp // WMMA
        ap = np.zeros((B, Mp, Kp), dtype=np.float16); ap[:, :M, :K] = a3.astype(np.float16)
        bp = np.zeros((B, Kp, Np), dtype=np.float16); bp[:, :K, :N] = b3.astype(np.float16)
        tta = lc.tensor(ap.reshape(-1), backend=be)     # fp16 in
        ttb = lc.tensor(bp.reshape(-1), backend=be)
        ttc = lc.empty(B * Mp * Np, backend=be)          # fp32 accumulate/out
        grid = (Np // WMMA, B * mtiles)                  # batch folded into grid dim 1
        _launcher(wmma_matmul_b.func, be)[grid, (32,)](tta, ttb, ttc, Mp, Np, Kp, mtiles)
        return ttc.numpy().reshape(B, Mp, Np)[:, :M, :N].astype(np.float32)
    except Exception:  # noqa: BLE001 — fall back to the scalar tiled kernel
        return None


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

        # CUDA tensor-core (wmma) fast path: fp16 inputs, fp32 accumulate.
        # Matches cuBLAS's hardware; kept as our own portable-ish kernel.
        if be == "cuda" and _TENSOR_CORES:
            r = _mm_wmma(a3, b3, B, M, N, K, be)
            if r is not None:
                return r.reshape(out_shape).astype(np.float32)

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


def matmul_resident(a, b):
    """Device-resident batched matmul over Tensors `a`, `b`. Reuses cached
    device buffers (so a shared activation is uploaded once) and keeps the
    result on the GPU. Returns (out_ndarray, out_device_buffer) or None if the
    fast path doesn't apply (caller falls back to `matmul`).

    16-aligned shapes only, so the cached raw buffers need no host padding —
    the transformer's common case.
    """
    if not _ENABLED or _BACKEND is None or not _RESIDENT:
        return None
    ad, bd = a.data, b.data
    if ad.ndim < 2 or bd.ndim < 2 or ad.dtype != np.float32 or bd.dtype != np.float32:
        return None
    if _BACKEND == "cuda" and _TENSOR_CORES:
        return None
    K = ad.shape[-1]
    N = bd.shape[-1]
    out_shape = (*ad.shape[:-1], N)
    if bd.ndim == 2:
        if bd.shape[0] != K:
            return None
        B = 1
        M = int(np.prod(ad.shape[:-1])) if ad.ndim > 1 else ad.shape[0]
    else:
        Bs = int(np.prod(ad.shape[:-2])) if ad.ndim > 2 else 1
        Bb = int(np.prod(bd.shape[:-2])) if bd.ndim > 2 else 1
        if Bs != Bb or bd.shape[-2] != K:
            return None
        B, M = Bs, ad.shape[-2]
    if M % 16 or N % 16 or K % 16:
        return None
    try:
        import locomp as lc
        from axis.accel.batched_tiled import TILE, tiled_matmul_b
        be = _BACKEND
        a_dev = to_device(a)
        b_dev = to_device(b)
        if a_dev is None or b_dev is None:
            return None
        mtiles = M // TILE
        nt = K // TILE
        out_dev = lc.empty(B * M * N, backend=be)
        grid = (N // TILE, B * mtiles)
        _launcher(tiled_matmul_b.func, be)[grid, (TILE, TILE)](
            a_dev, b_dev, out_dev, M, N, K, nt, TILE, mtiles)
        data = out_dev.numpy().reshape(out_shape).astype(np.float32)
        return data, out_dev
    except Exception:  # noqa: BLE001
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
