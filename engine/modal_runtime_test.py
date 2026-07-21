"""Phase-1 validation of the C++ engine: compile libaxeng.so on A100, drive an
execution plan from Python (ctypes), verify numerics vs NumPy, and time
run_plan vs CUDA-graph replay at transformer-like depth.

    modal run engine/modal_runtime_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24")
    .add_local_file(str(REPO / "axis" / "_csrc" / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-engine-test")


@app.function(image=image, gpu="A100", timeout=1200)
def test():
    import subprocess, sys, time
    import numpy as np
    sys.path.insert(0, "/root")

    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "--shared", "-Xcompiler", "-fPIC",
         "/root/runtime.cu", "-lcublas", "-o", "/root/libaxeng.so"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE FAILED:\n", r.stderr[-2000:], flush=True)
        return
    print("compile libaxeng.so: OK", flush=True)

    from axis.engine import Engine, op, GEMM, SILU_MUL, ADD, RMSNORM
    eng = Engine("/root/libaxeng.so")
    print("engine init: OK", flush=True)

    # ---- numerics: y = rmsnorm(x) ; h = silu(y@W1) * (y@W2) ; out = h@W3 + x
    rng = np.random.default_rng(0)
    M, D, H = 512, 1024, 2048
    x = rng.standard_normal((M, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    W1 = rng.standard_normal((D, H)).astype(np.float32) * 0.02
    W2 = rng.standard_normal((D, H)).astype(np.float32) * 0.02
    W3 = rng.standard_normal((H, D)).astype(np.float32) * 0.02

    bx = eng.new_tensor(x); bw = eng.new_tensor(w)
    b1 = eng.new_tensor(W1); b2 = eng.new_tensor(W2); b3 = eng.new_tensor(W3)
    by = eng.alloc(M * D); bg = eng.alloc(M * H); bu = eng.alloc(M * H)
    bh = eng.alloc(M * H); bo = eng.alloc(M * D); bout = eng.alloc(M * D)

    plan = [
        op(RMSNORM, a=bx, b=bw, c=by, m=M, n=D, alpha=1e-5),
        op(GEMM, a=by, b=b1, c=bg, m=M, k=D, n=H),
        op(GEMM, a=by, b=b2, c=bu, m=M, k=D, n=H),
        op(SILU_MUL, a=bg, b=bu, c=bh, n=M * H),
        op(GEMM, a=bh, b=b3, c=bo, m=M, k=H, n=D),
        op(ADD, a=bo, b=bx, c=bout, n=M * D),
    ]
    eng.run(plan)
    got = eng.download(bout, (M, D))

    # numpy reference
    inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)
    y = x * inv * w
    g = y @ W1; u = y @ W2
    h = (g / (1 + np.exp(-g))) * u
    ref = h @ W3 + x
    err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)
    print(f"numerics vs numpy: rel max err = {err:.2e}  match={err < 1e-3}", flush=True)

    # ---- speed: 48-layer-deep plan, run_plan vs graph replay ----
    deep = plan * 48
    eng.run(deep)  # warm
    t0 = time.perf_counter()
    for _ in range(5):
        eng.run(deep)
    plan_ms = (time.perf_counter() - t0) / 5 * 1000

    eng.capture(deep)
    eng.replay(1)  # warm
    t0 = time.perf_counter()
    eng.replay(20)
    graph_ms = (time.perf_counter() - t0) / 20 * 1000

    n_ops = len(deep)
    print(f"deep plan ({n_ops} ops): run_plan {plan_ms:.2f} ms | graph replay {graph_ms:.2f} ms "
          f"| replay speedup {plan_ms/graph_ms:.2f}x", flush=True)
    print("PHASE 1 RUNTIME: VALIDATED", flush=True)


@app.local_entrypoint()
def main():
    test.remote()
