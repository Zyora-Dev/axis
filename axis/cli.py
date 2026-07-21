"""axis.cli — the `axis-zyora` command line."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from axis import __version__
from axis._build import build_engine, engine_lib, detect_arch, _CSRC


def _check() -> None:
    print(f"axis-zyora {__version__}")
    nvcc = shutil.which("nvcc")
    print(f"  nvcc          : {nvcc or 'NOT FOUND (install CUDA toolkit to build the engine)'}")
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print(f"  gpu           : {gpu or 'none detected'}")
        print(f"  arch          : {detect_arch()}")
    else:
        print("  gpu           : nvidia-smi not found")
    src = _CSRC / "runtime.cu"
    print(f"  engine source : {'present' if src.exists() else 'MISSING'}  ({src})")
    try:
        print(f"  engine lib    : {engine_lib()}")
    except Exception as e:
        print(f"  engine lib    : not built ({e})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="axis-zyora",
        description="Axis — Zyora Labs GPU training & fine-tuning engine")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("version", help="print the version")
    sub.add_parser("check", help="check the environment (GPU, nvcc, engine)")
    b = sub.add_parser("build", help="compile the CUDA engine (libaxeng.so)")
    b.add_argument("--arch", default=None, help="nvcc arch, e.g. sm_90 (auto-detected)")
    b.add_argument("--nccl", action="store_true", help="build with NCCL (multi-GPU)")
    b.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = p.parse_args(argv)

    if args.cmd in (None, "version"):
        print(f"axis-zyora {__version__}")
        return 0
    if args.cmd == "check":
        _check()
        return 0
    if args.cmd == "build":
        try:
            path = build_engine(arch=args.arch, nccl=args.nccl, force=args.force)
        except Exception as e:
            print(f"build failed: {e}", file=sys.stderr)
            return 1
        print(f"engine built: {path}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
