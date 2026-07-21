"""HIP port build-check for the Axis runtime (#6).
Modal has no AMD GPU on this account, so we CANNOT run kernels — but we CAN
verify the ROCm toolchain compiles runtime.cu with -DAXIS_HIP (the port's
correctness bar until real MI300X time). Also re-confirms the CUDA build +
parity still pass after the portability guards (regression).

    modal run engine/modal_hip_build.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
rocm_image = (
    modal.Image.from_registry("rocm/dev-ubuntu-22.04:6.2-complete", add_python="3.11")
    .apt_install("rocwmma-dev", "hipblas-dev", "rccl-dev")
    .pip_install("numpy>=1.24")
    .add_local_file(str(REPO / "axis" / "_csrc" / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_file(str(REPO / "axis" / "_csrc" / "hip_compat.h"), remote_path="/root/hip_compat.h")
)
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24")
    .add_local_file(str(REPO / "axis" / "_csrc" / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-hip-build")


@app.function(image=rocm_image, timeout=1200)
def hip_build():
    import subprocess
    # tiled path only (no WMMA flash on HIP v1); +RCCL data parallel
    cmd = ["hipcc", "-O3", "--offload-arch=gfx942", "--shared", "-fPIC",
           "-DAXIS_HIP", "-DAXIS_NCCL", "/root/runtime.cu",
           "-lhipblas", "-lrccl", "-o", "/root/libaxeng_hip.so"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("HIP+RCCL build:", "OK" if r.returncode == 0 else "FAIL", flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:], flush=True)
        return
    import os
    sz = os.path.getsize("/root/libaxeng_hip.so")
    # symbol presence check
    nm = subprocess.run(["nm", "-D", "/root/libaxeng_hip.so"],
                        capture_output=True, text=True).stdout
    syms = [s for s in ("eng_init", "eng_run_plan", "eng_alloc", "eng_has_flash",
                        "eng_nccl_init", "eng_capture_plan") if s in nm]
    print(f"libaxeng_hip.so: {sz//1024} KB | exported: {', '.join(syms)}", flush=True)
    print(f"HIP BUILD: {'PASS' if len(syms) == 6 else 'FAIL'}", flush=True)


@app.function(image=cuda_image, gpu="A100", timeout=1200)
def cuda_regression():
    import subprocess, sys
    import numpy as np
    sys.path.insert(0, "/root")
    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "--shared", "-Xcompiler", "-fPIC",
         "/root/runtime.cu", "-lcublas", "-o", "/root/libaxeng.so"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("CUDA COMPILE FAILED:\n", r.stderr[-2000:], flush=True)
        return
    print("nvcc: OK (guards don't break CUDA)", flush=True)

    import axis
    from axis import nn
    from axis.tensor import Tensor
    from axis.compile import compile_model
    V, B, T, D, H, KV, MLP, L = 500, 2, 32, 128, 4, 2, 256, 3
    axis.manual_seed(0)
    m = nn.Transformer(vocab_size=V, dim=D, n_layers=L, n_heads=H, n_kv_heads=KV,
                       mlp_hidden=MLP, max_seq_len=T, tie_embeddings=True)
    rng = np.random.default_rng(0)
    toks = rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]
    lt = m.loss(Tensor(inp), Tensor(tgt)); rl = float(lt.data); lt.backward()
    rg = {n: p.grad.copy() for n, p in m.named_parameters()}

    def build():
        axis.manual_seed(0)
        return nn.Transformer(vocab_size=V, dim=D, n_layers=L, n_heads=H,
                              n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=T,
                              tie_embeddings=True)
    ok = True
    for dt_, impl, gtol in (("fp32", "tiled", 1e-3), ("bf16", "flash", 1.5e-1),
                            ("bf16", "tiled", 1.5e-1)):
        ct = compile_model(build(), B, T, lib_path="/root/libaxeng.so",
                           tf32=False, dtype=dt_, attn_impl=impl)
        gl = ct.step(inp, tgt, t=1); gg = ct.grads()
        worst = max(np.abs(gg[nm] - g).max() / (np.abs(g).max() + 1e-9)
                    for nm, g in rg.items())
        okx = abs(rl - gl) < (1e-4 if dt_ == "fp32" else 5e-2) and worst < gtol
        ok &= okx
        print(f"{dt_} {impl} [{ct.attn_impl}]: grad rel {worst:.2e} "
              f"-> {'PASS' if okx else 'FAIL'}", flush=True)
    print(f"CUDA REGRESSION: {'PASS' if ok else 'FAIL'}", flush=True)


@app.local_entrypoint()
def main():
    cuda_regression.remote()
    hip_build.remote()
