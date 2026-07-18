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
 EMBED, EMBED_BWD, CE, TICK) = range(22)


class _EngOp(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("a", ctypes.c_int), ("b", ctypes.c_int),
        ("c", ctypes.c_int), ("d", ctypes.c_int),
        ("m", ctypes.c_int), ("n", ctypes.c_int), ("k", ctypes.c_int),
        ("batch", ctypes.c_int), ("tb", ctypes.c_int),
        ("sa", ctypes.c_int), ("sb", ctypes.c_int), ("sc", ctypes.c_int),
        ("alpha", ctypes.c_float), ("beta", ctypes.c_float), ("gamma", ctypes.c_float),
    ]


def op(kind: int, a: int = -1, b: int = -1, c: int = -1, d: int = -1,
       m: int = 0, n: int = 0, k: int = 0,
       batch: int = 0, tb: int = 0, sa: int = 0, sb: int = 0, sc: int = 0,
       alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0) -> Tuple:
    """Op descriptor. Notable role maps:
    GEMM_SB:  tb=0: c=a@b; tb=1: c=a@b^T (b=[n,k]); tb=2: c=a[k,m]^T@b[k,n]
    PERM_0213: dims (m,n,k,batch) = (d0,d1,d2,d3), out [d0,d2,d1,d3]
    ROPE:     a=[batch*m rows, n]; b=cos, d=sin; tb=1 -> inverse (backward)
    SOFTMAX_CAUSAL/BWD: rows batch*m, width m; a(,b)->c
    REPEAT_KV(_BWD): batch=B, tb=KV, n=H, m=T, k=dh
    RMSNORM_BWD: a=x b=w d=g -> c=dx, tb=tmp buffer (colsum -> dw)
    SILU_BWD: a=g b=u d=grad -> c=dg, tb=du buffer
    EMBED(_BWD): a=table|g, b=ids(float) -> c; m=N n=D
    CE: a=logits b=targets -> c=dlogits d=loss[1]; m=N n=V
    ADAMW: a=p b=g c=m d=v; alpha=lr*sqrt(bc2)/bc1, beta=lr*wd, gamma=eps*sqrt(bc2)
    """
    return (kind, a, b, c, d, m, n, k, batch, tb, sa, sb, sc, alpha, beta, gamma)


class Engine:
    """Owns the runtime: buffers, plans, graphs."""

    def __init__(self, lib_path: str = "libaxeng.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.eng_init.restype = ctypes.c_int
        self.lib.eng_alloc.argtypes = [ctypes.c_int, ctypes.c_longlong]
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

    # ── buffers ──
    def alloc(self, nfloats: int) -> int:
        idx = self._next
        self._next += 1
        rc = self.lib.eng_alloc(idx, nfloats)
        if rc:
            raise RuntimeError(f"alloc({nfloats}) failed rc={rc}")
        return idx

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
            arr[i] = _EngOp(*[int(x) if j < 13 else float(x) for j, x in enumerate(o)])
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
