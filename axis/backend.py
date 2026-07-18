"""axis.backend — CPU/GPU array-module dispatch.

Axis runs the SAME engine on NumPy (CPU) or CuPy (GPU). CuPy is a drop-in
NumPy replacement whose matmul is cuBLAS and whose elementwise ops run on the
GPU — so moving a model to the GPU (`model.to_gpu()`) makes the whole forward
AND backward run on the device with no per-op host round-trip.

`array_module(*arrays)` returns cupy if any array is a CuPy array, else numpy,
so every op auto-dispatches to the right device based on where its data lives.
When CuPy isn't installed, everything stays on NumPy.
"""
from __future__ import annotations

import os

import numpy as _np

# Enable TF32 tensor cores for cuBLAS/cuDNN on Ampere+ (what PyTorch does by
# default). Same fp32 dtype and range, ~fp32 accuracy, faster matmul. Must be
# set before cupy imports. Opt out with AXIS_TF32=0.
os.environ["CUPY_TF32"] = os.environ.get("AXIS_TF32", "1")

try:
    import cupy as _cp
    import cupyx as _cpx
    _HAS_CUPY = True
except Exception:  # noqa: BLE001 — no GPU / cupy not installed
    _cp = None
    _cpx = None
    _HAS_CUPY = False


def has_cupy() -> bool:
    return _HAS_CUPY


# Active device for NEW tensors. When "gpu", Tensor construction moves data to
# the GPU so constants/batches/intermediates all live on the device (no mixing
# of numpy + cupy arrays, which cupy rejects).
_DEVICE = "cpu"


def set_device(d: str) -> None:
    global _DEVICE
    _DEVICE = d


def device() -> str:
    return _DEVICE


def is_gpu_array(a) -> bool:
    return _HAS_CUPY and isinstance(a, _cp.ndarray)


def array_module(*arrays):
    """Return cupy if any argument is a CuPy array, else numpy."""
    if _HAS_CUPY:
        for a in arrays:
            if isinstance(a, _cp.ndarray):
                return _cp
    return _np


def to_numpy(a):
    """Bring an array to host as a numpy array."""
    if is_gpu_array(a):
        return _cp.asnumpy(a)
    return _np.asarray(a)


def to_gpu_array(a):
    """Move a numpy array to the GPU (no-op if already cupy / no cupy)."""
    if not _HAS_CUPY:
        return a
    if isinstance(a, _cp.ndarray):
        return a
    return _cp.asarray(a)


def like(ref, x):
    """Return `x` on the same device as `ref`."""
    xp = array_module(ref)
    return xp.asarray(x)


def scatter_add(target, idx, values) -> None:
    """In-place target[idx] += values (device-aware scatter-add)."""
    if is_gpu_array(target):
        # cupy index arrays required for advanced indexing
        if is_gpu_array(idx) is False and not isinstance(idx, tuple):
            idx = _cp.asarray(idx)
        _cpx.scatter_add(target, idx, values)
    else:
        _np.add.at(target, idx, values)


class _XP:
    """Proxy that forwards `np.<fn>(arr, ...)` to the array-module of `arr`
    (cupy for GPU arrays, numpy otherwise). Type/constant attributes pass
    through to numpy unchanged (they interoperate with cupy arrays)."""

    _PASSTHROUGH = frozenset({
        "float32", "float64", "float16", "int64", "int32", "int16", "int8",
        "uint8", "uint32", "bool_", "newaxis", "inf", "nan", "pi", "e", "dtype",
    })

    def __getattr__(self, name):
        if name in self._PASSTHROUGH:
            return getattr(_np, name)

        def fn(*args, **kwargs):
            arrs = [a for a in args if isinstance(a, _np.ndarray)
                    or (_HAS_CUPY and isinstance(a, _cp.ndarray))]
            xp = array_module(*arrs) if arrs else _np
            return getattr(xp, name)(*args, **kwargs)

        return fn


# Import this as `np` inside ops so every np.<fn>(array) call auto-dispatches.
xp = _XP()
