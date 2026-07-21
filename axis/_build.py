"""axis._build — locate or compile the CUDA engine (libaxeng.so).

The C++/CUDA runtime source ships inside the package (axis/_csrc/runtime.cu).
This module compiles it with the local `nvcc` on demand and caches the result,
so `pip install axis-zyora` + a first `compile_model(...)` (or `axis-zyora build`)
produces a working engine without a manual nvcc step.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

_CSRC = pathlib.Path(__file__).parent / "_csrc"


def _cache_dir() -> pathlib.Path:
    d = pathlib.Path(os.environ.get(
        "AXIS_CACHE", pathlib.Path.home() / ".cache" / "axis-zyora"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def detect_arch() -> str:
    """GPU compute capability as an nvcc -arch string (e.g. sm_80). Override
    with AXIS_ARCH; falls back to sm_80 if no GPU is visible."""
    a = os.environ.get("AXIS_ARCH")
    if a:
        return a
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True).stdout
        cc = out.strip().splitlines()[0].strip().replace(".", "")
        return f"sm_{cc}"
    except Exception:
        return "sm_80"


def _nccl_paths():
    try:
        import nvidia.nccl
        base = pathlib.Path(list(nvidia.nccl.__path__)[0])
        return base / "include", base / "lib"
    except Exception:
        return None, None


def build_engine(arch: str = None, nccl: bool = False,
                 out: str = None, force: bool = False) -> str:
    """Compile axis/_csrc/runtime.cu -> libaxeng.so and return its path.
    Cached under ~/.cache/axis-zyora (or $AXIS_CACHE)."""
    arch = arch or detect_arch()
    src = _CSRC / "runtime.cu"
    if not src.exists():
        raise FileNotFoundError(f"engine source not found: {src}")
    if out is None:
        out = _cache_dir() / f"libaxeng-{arch}{'-nccl' if nccl else ''}.so"
    out = pathlib.Path(out)
    if out.exists() and not force:
        return str(out)
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            "nvcc not found — install the NVIDIA CUDA toolkit to build the "
            "Axis engine, or set AXIS_ENGINE_LIB to a prebuilt libaxeng.so")
    cmd = [nvcc, "-O3", f"-arch={arch}", "--shared", "-Xcompiler", "-fPIC", str(src)]
    if nccl:
        inc, lib = _nccl_paths()
        cmd += ["-DAXIS_NCCL"]
        if inc:
            cmd += [f"-I{inc}"]
        if lib:
            cmd += [f"-L{lib}"]
        cmd += ["-lnccl"]
    cmd += ["-lcublas", "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("Axis engine build failed:\n" + r.stderr[-3000:])
    return str(out)


def engine_lib(arch: str = None, nccl: bool = False) -> str:
    """Resolve a usable engine: $AXIS_ENGINE_LIB if set, else cached, else built."""
    env = os.environ.get("AXIS_ENGINE_LIB")
    if env and pathlib.Path(env).exists():
        return env
    return build_engine(arch=arch, nccl=nccl)
