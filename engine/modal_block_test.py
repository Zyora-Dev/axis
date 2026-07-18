"""Phase 2a: FULL TransformerBlock forward, lowered to the C++ engine, vs the
eager Axis oracle. Validates every new runtime op (batched GEMM, RoPE, causal
softmax, permutes, GQA repeat) against the ground-truth engine.

    modal run engine/modal_block_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24")
    .add_local_file(str(HERE / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-engine-block")

B, T, D, H, KV, DH, MLP = 2, 128, 512, 8, 4, 64, 1024


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
    print("compile: OK", flush=True)

    import axis
    from axis import nn
    from axis.tensor import Tensor
    from axis.engine import (Engine, op, GEMM, ADD, RMSNORM, SILU_MUL,
                             GEMM_SB, PERM_0213, ROPE, SOFTMAX_CAUSAL, REPEAT_KV)

    # ---- eager oracle ----
    axis.manual_seed(0)
    blk = nn.TransformerBlock(dim=D, n_heads=H, n_kv_heads=KV, mlp_hidden=MLP,
                              max_seq_len=T)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, T, D)).astype(np.float32)
    ref = blk(Tensor(x)).numpy()
    cos, sin = blk.attn._cos, blk.attn._sin       # [T, DH/2]
    p = dict(blk.named_parameters())
    scale = 1.0 / np.sqrt(DH)

    # ---- lower to engine plan ----
    eng = Engine("/root/libaxeng.so")
    N = B * T
    bx = eng.new_tensor(x.reshape(N, D))
    b_n1w = eng.new_tensor(p["attn_norm.weight"].data)
    b_n2w = eng.new_tensor(p["mlp_norm.weight"].data)
    b_wq = eng.new_tensor(p["attn.q_proj.weight"].data)   # [D, H*DH]
    b_wk = eng.new_tensor(p["attn.k_proj.weight"].data)   # [D, KV*DH]
    b_wv = eng.new_tensor(p["attn.v_proj.weight"].data)
    b_wo = eng.new_tensor(p["attn.o_proj.weight"].data)   # [H*DH, D]
    b_wg = eng.new_tensor(p["mlp.gate_proj.weight"].data)
    b_wu = eng.new_tensor(p["mlp.up_proj.weight"].data)
    b_wd = eng.new_tensor(p["mlp.down_proj.weight"].data)
    b_cos = eng.new_tensor(cos)
    b_sin = eng.new_tensor(sin)

    y = eng.alloc(N * D)
    q0 = eng.alloc(N * H * DH); k0 = eng.alloc(N * KV * DH); v0 = eng.alloc(N * KV * DH)
    q1 = eng.alloc(N * H * DH); k1 = eng.alloc(N * KV * DH); v1 = eng.alloc(N * KV * DH)
    q2 = eng.alloc(N * H * DH); k2 = eng.alloc(N * KV * DH)
    kr = eng.alloc(N * H * DH); vr = eng.alloc(N * H * DH)
    sc = eng.alloc(B * H * T * T); pr = eng.alloc(B * H * T * T)
    at = eng.alloc(N * H * DH); at2 = eng.alloc(N * H * DH)
    o = eng.alloc(N * D); r1 = eng.alloc(N * D)
    z = eng.alloc(N * D); g = eng.alloc(N * MLP); u = eng.alloc(N * MLP)
    hbuf = eng.alloc(N * MLP); mlp_o = eng.alloc(N * D); out = eng.alloc(N * D)

    plan = [
        op(RMSNORM, a=bx, b=b_n1w, c=y, m=N, n=D, alpha=1e-5),
        op(GEMM, a=y, b=b_wq, c=q0, m=N, k=D, n=H * DH),
        op(GEMM, a=y, b=b_wk, c=k0, m=N, k=D, n=KV * DH),
        op(GEMM, a=y, b=b_wv, c=v0, m=N, k=D, n=KV * DH),
        # [B,T,H,DH] -> [B,H,T,DH]
        op(PERM_0213, a=q0, c=q1, m=B, n=T, k=H, batch=DH),
        op(PERM_0213, a=k0, c=k1, m=B, n=T, k=KV, batch=DH),
        op(PERM_0213, a=v0, c=v1, m=B, n=T, k=KV, batch=DH),
        # rope on q,k
        op(ROPE, a=q1, b=b_cos, d=b_sin, c=q2, batch=B * H, m=T, n=DH),
        op(ROPE, a=k1, b=b_cos, d=b_sin, c=k2, batch=B * KV, m=T, n=DH),
        # GQA repeat kv -> H heads
        op(REPEAT_KV, a=k2, c=kr, batch=B, tb=KV, n=H, m=T, k=DH),
        op(REPEAT_KV, a=v1, c=vr, batch=B, tb=KV, n=H, m=T, k=DH),
        # scores = q @ k^T * scale   [B*H, T, T]
        op(GEMM_SB, a=q2, b=kr, c=sc, m=T, k=DH, n=T, batch=B * H, tb=1,
           sa=T * DH, sb=T * DH, sc=T * T, alpha=scale),
        op(SOFTMAX_CAUSAL, a=sc, c=pr, batch=B * H, m=T),
        # attn = probs @ v   [B*H, T, DH]
        op(GEMM_SB, a=pr, b=vr, c=at, m=T, k=T, n=DH, batch=B * H,
           sa=T * T, sb=T * DH, sc=T * DH),
        # [B,H,T,DH] -> [B,T,H,DH] (flatten = [N, H*DH])
        op(PERM_0213, a=at, c=at2, m=B, n=H, k=T, batch=DH),
        op(GEMM, a=at2, b=b_wo, c=o, m=N, k=H * DH, n=D),
        op(ADD, a=bx, b=o, c=r1, n=N * D),
        # mlp
        op(RMSNORM, a=r1, b=b_n2w, c=z, m=N, n=D, alpha=1e-5),
        op(GEMM, a=z, b=b_wg, c=g, m=N, k=D, n=MLP),
        op(GEMM, a=z, b=b_wu, c=u, m=N, k=D, n=MLP),
        op(SILU_MUL, a=g, b=u, c=hbuf, n=N * MLP),
        op(GEMM, a=hbuf, b=b_wd, c=mlp_o, m=N, k=MLP, n=D),
        op(ADD, a=r1, b=mlp_o, c=out, n=N * D),
    ]
    eng.run(plan)
    got = eng.download(out, (B, T, D)).reshape(B, T, D)

    err = np.abs(got - ref).max()
    rel = err / (np.abs(ref).max() + 1e-9)
    print(f"BLOCK PARITY vs eager oracle: max|Δ|={err:.3e} rel={rel:.3e} "
          f"match={np.allclose(got, ref, rtol=2e-3, atol=2e-3)}", flush=True)

    # speed: 48 blocks deep, run + graph replay
    deep = plan * 48
    eng.run(deep)
    t0 = time.perf_counter()
    for _ in range(5):
        eng.run(deep)
    plan_ms = (time.perf_counter() - t0) / 5 * 1000
    eng.capture(deep)
    eng.replay(1)
    t0 = time.perf_counter()
    eng.replay(20)
    graph_ms = (time.perf_counter() - t0) / 20 * 1000
    toks = B * T
    print(f"48-block fwd ({len(deep)} ops): plan {plan_ms:.1f} ms | graph {graph_ms:.1f} ms "
          f"| fwd tok/s (graph) {toks/(graph_ms/1000):,.0f}", flush=True)
    print("PHASE 2a: BLOCK FORWARD VALIDATED", flush=True)


@app.local_entrypoint()
def main():
    test.remote()
