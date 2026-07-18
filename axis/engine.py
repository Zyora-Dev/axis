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
GEMM, ADD, MUL, SILU_MUL, RMSNORM, ADAMW, SCALE, COPY = range(8)


class _EngOp(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("a", ctypes.c_int), ("b", ctypes.c_int),
        ("c", ctypes.c_int), ("d", ctypes.c_int),
        ("m", ctypes.c_int), ("n", ctypes.c_int), ("k", ctypes.c_int),
        ("alpha", ctypes.c_float), ("beta", ctypes.c_float),
    ]


def op(kind: int, a: int = -1, b: int = -1, c: int = -1, d: int = -1,
       m: int = 0, n: int = 0, k: int = 0,
       alpha: float = 0.0, beta: float = 0.0) -> Tuple:
    return (kind, a, b, c, d, m, n, k, alpha, beta)


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
            arr[i] = _EngOp(*[int(x) if j < 8 else float(x) for j, x in enumerate(o)])
        return arr

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
