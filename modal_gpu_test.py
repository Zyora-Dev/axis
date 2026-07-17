"""Run Axis on a real NVIDIA A100 via Modal.

Installs locomp (CUDA backend) + torch, mounts the Axis source, runs the full
test suite on GPU, and benchmarks the tiled matmul + fused attention against
NumPy and PyTorch — the honest datacenter-GPU numbers.

    modal run modal_gpu_test.py
"""
import pathlib

import modal

REPO = pathlib.Path(__file__).parent

# CUDA *devel* base image ships nvcc — locomp compiles CUDA C with it.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "locomp==1.0.0", "pytest>=7.0", "torch")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
    .add_local_dir(str(REPO / "tests"), remote_path="/root/tests")
)

app = modal.App("axis-gpu-test", image=image)


@app.function(gpu="A100", timeout=1800)
def run():
    import os
    import subprocess
    import sys
    import time

    import numpy as np

    os.chdir("/root")
    sys.path.insert(0, "/root")

    print("=" * 70)
    print("AXIS on NVIDIA A100 (Modal) — CUDA backend via locomp")
    print("=" * 70)

    # ── GPU / backend sanity ────────────────────────────────────────────
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    print("nvcc:", subprocess.run(["bash", "-c", "which nvcc && nvcc --version | tail -2"],
                                  capture_output=True, text=True).stdout.strip())

    # Diagnostic: probe locomp CUDA directly and surface any error.
    try:
        import locomp as lc
        x = lc.tensor(np.arange(4, dtype=np.float32), backend="cuda")
        print("locomp cuda tensor ok:", x.numpy())
    except Exception as e:
        import traceback
        print("locomp cuda probe FAILED:")
        traceback.print_exc()

    from axis import accel
    print("locomp GPU available:", accel.available())
    accel.enable()

    # ── Full test suite on GPU ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST SUITE (GPU)")
    print("=" * 70)
    rc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                         "--no-header", "-p", "no:cacheprovider"],
                        capture_output=True, text=True)
    print(rc.stdout[-3000:])
    if rc.returncode != 0:
        print("STDERR:", rc.stderr[-2000:])

    # ── Benchmarks ──────────────────────────────────────────────────────
    import math
    from axis.tensor import Tensor
    from axis import ops

    try:
        import torch
        has_torch = torch.cuda.is_available()
    except Exception:
        has_torch = False

    def bench(fn, n=30, warmup=5):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000  # ms

    print("\n" + "=" * 70)
    print("MATMUL BENCHMARK  (batched [B,M,K]@[B,K,N])")
    print("=" * 70)
    for (B, M, K, N) in [(8, 512, 512, 512), (16, 1024, 1024, 1024), (32, 2048, 512, 2048)]:
        a = np.random.randn(B, M, K).astype(np.float32)
        b = np.random.randn(B, K, N).astype(np.float32)
        accel.enable()
        out = accel.matmul(a, b)
        ok = out is not None and np.abs(out - a @ b).max() < 1e-1
        t_gpu = bench(lambda: accel.matmul(a, b))
        accel.disable()
        t_np = bench(lambda: a @ b, n=10)
        line = f"[{B}x{M}x{K}@{K}x{N}]  locomp-tiled {t_gpu:8.2f} ms | numpy {t_np:8.2f} ms | correct={ok}"
        if has_torch:
            ta = torch.tensor(a, device="cuda")
            tb = torch.tensor(b, device="cuda")
            def tm():
                torch.cuda.synchronize(); r = ta @ tb; torch.cuda.synchronize(); return r
            line += f" | torch-cuda {bench(tm):8.2f} ms"
        print(line)

    print("\n" + "=" * 70)
    print("FUSED CAUSAL ATTENTION BENCHMARK  [B,H,T,D]")
    print("=" * 70)
    for (B, H, T, D) in [(2, 8, 128, 64), (4, 16, 256, 64), (8, 16, 512, 64)]:
        q = (np.random.randn(B, H, T, D) * 0.5).astype(np.float32)
        k = (np.random.randn(B, H, T, D) * 0.5).astype(np.float32)
        v = (np.random.randn(B, H, T, D) * 0.5).astype(np.float32)
        scale = 1.0 / math.sqrt(D)
        accel.enable()
        res = accel.fused_causal_attention(q, k, v, scale)
        ok = res is not None
        t_gpu = bench(lambda: accel.fused_causal_attention(q, k, v, scale), n=20)
        accel.disable()
        t_cpu = bench(lambda: accel.fused_causal_attention(q, k, v, scale), n=5)
        print(f"[{B}x{H}x{T}x{D}]  locomp-fused {t_gpu:8.2f} ms | numpy {t_cpu:8.2f} ms | executed={ok}")

    print("\n" + "=" * 70)
    print("END-TO-END: train a tiny transformer on A100 (GPU kernels in loop)")
    print("=" * 70)
    import axis
    from axis import nn, optim
    axis.manual_seed(0)
    accel.enable()
    vocab, seq = 256, 64
    model = nn.Transformer(vocab_size=vocab, dim=256, n_layers=4, n_heads=8,
                           n_kv_heads=4, mlp_hidden=512, max_seq_len=seq)
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    toks = rng.integers(0, vocab, size=(8, seq)).astype(np.int64)
    inp, tgt = Tensor(toks[:, :-1]), Tensor(toks[:, 1:])
    print(f"params: {model.num_parameters():,}")
    t0 = time.perf_counter()
    for step in range(20):
        model.zero_grad()
        loss = model.loss(inp, tgt)
        loss.backward()
        optim.clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == 19:
            print(f"  step {step:2d}  loss {loss.item():.4f}")
    print(f"  20 steps in {time.perf_counter()-t0:.1f}s")

    print("\nDONE.")
