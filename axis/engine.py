"""axis.engine — ctypes binding to the C++/CUDA runtime (libaxeng.so).

The runtime executes a whole training step as an execution PLAN (op
descriptors over a buffer table) with its own cuBLAS handle, and can capture
the plan into a CUDA graph for zero-Python replay.

Dependency-light on purpose: plain ctypes, no pybind.
"""
from __future__ import annotations

import ctypes
from typing import List, Sequence, Tuple

import numpy as np

# op kinds — must match runtime.cu
(GEMM, ADD, MUL, SILU_MUL, RMSNORM, ADAMW, SCALE, COPY,
 GEMM_SB, PERM_0213, ROPE, SOFTMAX_CAUSAL, REPEAT_KV,
 RMSNORM_BWD, COLSUM, REPEAT_KV_BWD, SOFTMAX_BWD, SILU_BWD,
 EMBED, EMBED_BWD, CE, TICK, CAST, FLASH, ROWDOT, FLASH_BWD,
 ALLREDUCE, GROUP, L2ACC, CLIPSCALE, COUNTVALID) = range(31)


class _EngOp(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("a", ctypes.c_int), ("b", ctypes.c_int),
        ("c", ctypes.c_int), ("d", ctypes.c_int),
        ("m", ctypes.c_int), ("n", ctypes.c_int), ("k", ctypes.c_int),
        ("batch", ctypes.c_int), ("tb", ctypes.c_int),
        ("sa", ctypes.c_int), ("sb", ctypes.c_int), ("sc", ctypes.c_int),
        ("dt", ctypes.c_int),
        ("oa", ctypes.c_int), ("ob", ctypes.c_int), ("oc", ctypes.c_int),
        ("alpha", ctypes.c_float), ("beta", ctypes.c_float), ("gamma", ctypes.c_float),
    ]


def op(kind: int, a: int = -1, b: int = -1, c: int = -1, d: int = -1,
       m: int = 0, n: int = 0, k: int = 0,
       batch: int = 0, tb: int = 0, sa: int = 0, sb: int = 0, sc: int = 0,
       dt: int = 0, oa: int = 0, ob: int = 0, oc: int = 0,
       alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0) -> Tuple:
    """Op descriptor. dt: 0=fp32, 1=bf16 storage, 2=bf16 inputs/fp32 output.
    oa/ob/oc: element offsets into a/b/c (GEMM query tiling).
    beta (GEMM): 0 = overwrite C, 1 = accumulate into C.
    Notable role maps:
    GEMM_SB:  tb=0: c=a@b; tb=1: c=a@b^T (b=[n,k]); tb=2: c=a[k,m]^T@b[k,n]
    PERM_0213: dims (m,n,k,batch) = (d0,d1,d2,d3), out [d0,d2,d1,d3]
    ROPE:     a=[batch*m rows, n]; b=cos, d=sin (fp32); tb=1 -> inverse
    SOFTMAX_CAUSAL: rows batch*m; untiled width=m; tiled: n=key range, k=q offset
    SOFTMAX_BWD: rows batch*m, width = n if n>0 else m
    REPEAT_KV(_BWD): batch=B, tb=KV, n=H, m=T, k=dh
    RMSNORM_BWD: a=x b=w d=g -> c=dx, tb=tmp buffer (colsum -> dw)
    SILU_BWD: a=g b=u d=grad -> c=dg, tb=du buffer
    EMBED(_BWD): a=table|g, b=ids(fp32) -> c; m=N n=D (BWD: dt>=1 means g bf16)
    CE: a=logits b=targets -> c=dlogits d=loss(fp32); m=N n=V; sa=denom buf
        (mean over valid tokens, 0->1/N); target<0 = ignored position
    ADAMW: a=master(fp32) b=g(fp32) c=m d=v; tb=t-buffer (-1 folded);
           sa=bf16 param mirror (0 = none); sb=device-lr buf (0->alpha const);
           oa=grad-scale buf (0->1); alpha=lr beta=wd(raw) gamma=eps
    CAST: a->c; tb=0 fp32->bf16, tb=1 bf16->fp32; m(*n)=count
    L2ACC: a=fp32 grad -> c=sumsq acc (atomicAdd, zero c first); n=count
    CLIPSCALE: a=sumsq acc -> c=scale (min(1, max_norm/(||g||+eps))); alpha=max_norm
    COUNTVALID: a=targets(fp32,<0 ignore) -> c=count acc (zero first); m=N
    FLASH: fused attention fwd (bf16): a=q b=k d=v c=o; m=T n=DH k=KV
           batch=B tb=H; sa=lse buffer (0=none); alpha=scale
    ROWDOT: c[r] = sum_d a[r,d]*b[r,d]; m=rows n=dim; fp32 out
    FLASH_BWD: a=q b=k d=v c=dO; sa=lse sb=D tb=dqf sc=dkf oa=dvf ob=H;
           m=T n=DH k=KV batch=B; alpha=scale; dq/dk/dv fp32, pre-zeroed
    ALLREDUCE: a=fp32 buffer, n=count — NCCL average across ranks
    GROUP: tb=0 ncclGroupStart, tb=1 ncclGroupEnd
    """
    return (kind, a, b, c, d, m, n, k, batch, tb, sa, sb, sc, dt, oa, ob, oc,
            alpha, beta, gamma)


class Engine:
    """Owns the runtime: buffers, plans, graphs."""

    def __init__(self, lib_path: str = "libaxeng.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.eng_init.restype = ctypes.c_int
        self.lib.eng_alloc.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_int]
        self.lib.eng_upload.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_longlong]
        self.lib.eng_download.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_longlong]
        self.lib.eng_run_plan.argtypes = [ctypes.POINTER(_EngOp), ctypes.c_int, ctypes.c_int]
        self.lib.eng_capture_plan.argtypes = [ctypes.POINTER(_EngOp), ctypes.c_int]
        self.lib.eng_replay.argtypes = [ctypes.c_int, ctypes.c_int]
        rc = self.lib.eng_init()
        if rc:
            raise RuntimeError(f"engine init failed rc={rc}")
        self._next = 0
        # backend capability: fused flash attention (CUDA WMMA; not HIP v1)
        try:
            self.has_flash = bool(self.lib.eng_has_flash())
        except AttributeError:
            self.has_flash = False

    # ── buffers ──
    def alloc(self, nelems: int, itemsize: int = 4) -> int:
        idx = self._next
        self._next += 1
        rc = self.lib.eng_alloc(idx, nelems, itemsize)
        if rc:
            raise RuntimeError(f"alloc({nelems}x{itemsize}) failed rc={rc}")
        return idx

    # ── NCCL (requires runtime built with -DAXIS_NCCL) ──
    def nccl_id(self) -> bytes:
        buf = (ctypes.c_char * 128)()
        rc = self.lib.eng_nccl_id(buf)
        if rc:
            raise RuntimeError(f"nccl_id failed rc={rc}")
        return bytes(buf)

    def nccl_init(self, rank: int, world: int, uid: bytes) -> None:
        rc = self.lib.eng_nccl_init(rank, world, uid)
        if rc:
            raise RuntimeError(f"nccl_init(rank={rank}) failed rc={rc}")

    def upload(self, idx: int, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        rc = self.lib.eng_upload(idx, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a.size)
        if rc:
            raise RuntimeError(f"upload failed rc={rc}")

    def download(self, idx: int, shape) -> np.ndarray:
        out = np.empty(shape, dtype=np.float32)
        rc = self.lib.eng_download(idx, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size)
        if rc:
            raise RuntimeError(f"download failed rc={rc}")
        return out

    def new_tensor(self, arr: np.ndarray) -> int:
        idx = self.alloc(arr.size)
        self.upload(idx, arr)
        return idx

    # ── plans ──
    def _pack(self, plan: Sequence[Tuple]):
        arr = (_EngOp * len(plan))()
        for i, o in enumerate(plan):
            arr[i] = _EngOp(*[int(x) if j < 17 else float(x) for j, x in enumerate(o)])
        return arr

    def zero(self, idx: int, nfloats: int) -> Tuple:
        """Plan op that zeroes a buffer (scale by 0 into itself)."""
        return op(SCALE, a=idx, c=idx, n=nfloats, alpha=0.0)

    def run(self, plan: Sequence[Tuple], sync: bool = True) -> None:
        arr = self._pack(plan)
        rc = self.lib.eng_run_plan(arr, len(plan), 1 if sync else 0)
        if rc:
            raise RuntimeError(f"run_plan failed rc={rc} (op {rc - 1000 if rc >= 1000 else '?'})")

    def capture(self, plan: Sequence[Tuple]) -> None:
        arr = self._pack(plan)
        rc = self.lib.eng_capture_plan(arr, len(plan))
        if rc:
            raise RuntimeError(f"capture failed rc={rc}")

    def replay(self, times: int = 1, sync: bool = True) -> None:
        rc = self.lib.eng_replay(times, 1 if sync else 0)
        if rc:
            raise RuntimeError(f"replay failed rc={rc}")

    def sync(self) -> None:
        self.lib.eng_sync()

    def set_tf32(self, on: bool) -> None:
        self.lib.eng_set_tf32(1 if on else 0)
