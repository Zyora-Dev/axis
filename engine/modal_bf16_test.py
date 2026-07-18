"""bf16 engine validation:
1) fp32 regression — the runtime was rewritten (templated kernels, GemmEx);
   the existing 3-block parity gate must still pass bit-tight.
2) bf16 parity — same 3-block model vs the eager fp32 oracle, relaxed
   tolerance (bf16 storage ~3 decimal digits).
3) loss-curve equivalence — 30 steps bf16 vs fp32 engine, curves must track.
4) 1B-class benchmark, seq 2048 — torch (bf16 autocast + SDPA) vs Axis fp32
   vs Axis bf16, one process per config (engine buffers are process-global).

    modal run engine/modal_bf16_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "torch")
    .add_local_file(str(HERE / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-engine-bf16")

# 1B-class benchmark shape (same as the long-seq test)
BV, BB, BT, BD, BH, BKV, BMLP, BL = 32000, 4, 2048, 1536, 24, 8, 4096, 48


def _compile_runtime():
    import subprocess
    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "--shared", "-Xcompiler", "-fPIC",
         "/root/runtime.cu", "-lcublas", "-o", "/root/libaxeng.so"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE FAILED:\n", r.stderr[-3000:], flush=True)
        raise SystemExit(1)
    print("nvcc: OK", flush=True)


def _bench_weights(rng):
    import numpy as np
    shapes = {"embed.weight": (BV, BD), "norm.weight": (BD,)}
    for i in range(BL):
        p = f"blocks.{i}."
        shapes.update({p + "attn_norm.weight": (BD,), p + "mlp_norm.weight": (BD,),
                       p + "attn.q_proj.weight": (BD, BH * (BD // BH)),
                       p + "attn.k_proj.weight": (BD, BKV * (BD // BH)),
                       p + "attn.v_proj.weight": (BD, BKV * (BD // BH)),
                       p + "attn.o_proj.weight": (BH * (BD // BH), BD),
                       p + "mlp.gate_proj.weight": (BD, BMLP),
                       p + "mlp.up_proj.weight": (BD, BMLP),
                       p + "mlp.down_proj.weight": (BMLP, BD)})
    return {nm: (np.ones(sh, dtype=np.float32) if "norm" in nm else
                 (rng.standard_normal(sh) * 0.02).astype(np.float32))
            for nm, sh in shapes.items()}


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def parity():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    _compile_runtime()

    import axis
    from axis import nn
    from axis.tensor import Tensor
    from axis.compile import CompiledTransformer

    V, B, T, D, H, KV, MLP, L = 500, 2, 32, 128, 4, 2, 256, 3
    axis.manual_seed(0)
    model = nn.Transformer(vocab_size=V, dim=D, n_layers=L, n_heads=H,
                           n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=T,
                           tie_embeddings=True)
    weights = {n: p.data.copy() for n, p in model.named_parameters()}
    rng = np.random.default_rng(0)
    toks = rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]

    loss_t = model.loss(Tensor(inp), Tensor(tgt))
    ref_loss = float(loss_t.data)
    loss_t.backward()
    ref_grads = {n: p.grad.copy() for n, p in model.named_parameters()}
    cfg = dict(vocab_size=V, dim=D, n_layers=L, n_heads=H, n_kv_heads=KV,
               mlp_hidden=MLP, tie_embeddings=True)

    def check(dtype, loss_tol, grad_tol):
        ct = CompiledTransformer("/root/libaxeng.so", cfg, weights, B, T,
                                 tf32=False, dtype=dtype)
        got_loss = ct.step(inp, tgt, t=1)
        got = ct.grads()
        worst = ("", 0.0)
        for nm, g in ref_grads.items():
            rel = np.abs(got[nm] - g).max() / (np.abs(g).max() + 1e-9)
            if rel > worst[1]:
                worst = (nm, rel)
        ok = abs(ref_loss - got_loss) < loss_tol and worst[1] < grad_tol
        print(f"{dtype} parity: loss Δ {abs(ref_loss-got_loss):.2e} | "
              f"worst grad rel {worst[1]:.2e} ({worst[0]}) -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        return ok

    ok32 = check("fp32", 1e-4, 1e-3)          # regression gate — bit-tight
    okbf = check("bf16", 5e-2, 1.5e-1)        # bf16 storage — relaxed
    if not ok32:
        print("FP32 REGRESSION — STOP", flush=True)
        return

    # ---- loss-curve equivalence: 30 steps, bf16 must track fp32 ----
    V2, B2, T2, D2, H2, KV2, MLP2, L2 = 1000, 4, 64, 256, 8, 4, 512, 4
    axis.manual_seed(1)
    m2 = nn.Transformer(vocab_size=V2, dim=D2, n_layers=L2, n_heads=H2,
                        n_kv_heads=KV2, mlp_hidden=MLP2, max_seq_len=T2,
                        tie_embeddings=True)
    w2 = {n: p.data.copy() for n, p in m2.named_parameters()}
    cfg2 = dict(vocab_size=V2, dim=D2, n_layers=L2, n_heads=H2, n_kv_heads=KV2,
                mlp_hidden=MLP2, tie_embeddings=True)
    r2 = np.random.default_rng(7)
    batches = [r2.integers(0, V2, size=(B2, T2 + 1)).astype(np.int64) for _ in range(30)]

    def curve(dtype):
        ct = CompiledTransformer("/root/libaxeng.so", cfg2, w2, B2, T2,
                                 lr=1e-3, tf32=True, dtype=dtype)
        return [ct.step(b[:, :-1], b[:, 1:]) for b in batches]

    c32 = curve("fp32")
    cbf = curve("bf16")
    drift = max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(c32, cbf))
    print("step   fp32     bf16", flush=True)
    for i in (0, 4, 9, 19, 29):
        print(f"{i+1:4d} {c32[i]:8.4f} {cbf[i]:8.4f}", flush=True)
    curve_ok = drift < 0.03
    print(f"loss-curve max rel drift over 30 steps: {drift:.4f} -> "
          f"{'PASS' if curve_ok else 'FAIL'}", flush=True)
    print(f"\nBF16 VALIDATION: {'PASS' if (okbf and curve_ok) else 'FAIL'}", flush=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600, single_use_containers=True)
def bench(mode: str):
    """One config per FRESH container (max_inputs=1 — torch's caching allocator
    would otherwise hoard the GPU across warm-container reuse). mode: torch|fp32|bf16."""
    import subprocess, sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    N = BB * BT
    rng = np.random.default_rng(1)
    toks = rng.integers(0, BV, size=(BB, BT + 1)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]

    def used_mb():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.strip().splitlines()[0])

    if mode == "torch":
        import torch
        import torch.nn as tnn
        import torch.nn.functional as F
        torch.backends.cuda.matmul.allow_tf32 = True
        DH = BD // BH

        class TRms(tnn.Module):
            def __init__(s, d):
                super().__init__(); s.w = tnn.Parameter(torch.ones(d))
            def forward(s, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * s.w

        class TBlock(tnn.Module):
            def __init__(s):
                super().__init__()
                s.n1, s.n2 = TRms(BD), TRms(BD)
                s.q = tnn.Linear(BD, BH * DH, bias=False)
                s.k = tnn.Linear(BD, BKV * DH, bias=False)
                s.v = tnn.Linear(BD, BKV * DH, bias=False)
                s.o = tnn.Linear(BH * DH, BD, bias=False)
                s.g = tnn.Linear(BD, BMLP, bias=False)
                s.u = tnn.Linear(BD, BMLP, bias=False)
                s.dn = tnn.Linear(BMLP, BD, bias=False)
            def forward(s, x, cos, sin):
                Bx, Tx, _ = x.shape
                y = s.n1(x)
                q = s.q(y).view(Bx, Tx, BH, DH).transpose(1, 2)
                k = s.k(y).view(Bx, Tx, BKV, DH).transpose(1, 2)
                v = s.v(y).view(Bx, Tx, BKV, DH).transpose(1, 2)
                def rope(t):
                    h = DH // 2
                    t1, t2 = t[..., :h], t[..., h:]
                    return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], -1)
                q, k = rope(q), rope(k)
                k = k.repeat_interleave(BH // BKV, dim=1)
                v = v.repeat_interleave(BH // BKV, dim=1)
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                a = a.transpose(1, 2).reshape(Bx, Tx, BH * DH)
                x = x + s.o(a)
                z = s.n2(x)
                return x + s.dn(F.silu(s.g(z)) * s.u(z))

        class TModel(tnn.Module):
            def __init__(s):
                super().__init__()
                s.emb = tnn.Embedding(BV, BD)
                s.blocks = tnn.ModuleList([TBlock() for _ in range(BL)])
                s.norm = TRms(BD)
                half = DH // 2
                fr = 1.0 / (10000.0 ** (torch.arange(half) / half))
                ang = torch.outer(torch.arange(BT), fr)
                s.register_buffer("cos", ang.cos()[None, None])
                s.register_buffer("sin", ang.sin()[None, None])
            def forward(s, ids):
                x = s.emb(ids)
                for b in s.blocks:
                    x = b(x, s.cos, s.sin)
                return s.norm(x) @ s.emb.weight.T

        tm = TModel().cuda().float()
        print(f"params: {sum(p.numel() for p in tm.parameters())/1e9:.2f}B", flush=True)
        topt = torch.optim.AdamW(tm.parameters(), lr=3e-4, betas=(0.9, 0.95),
                                 weight_decay=0.1)
        ti = torch.tensor(inp).cuda(); tt = torch.tensor(tgt).cuda()

        def step():
            topt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = tm(ti)
                loss = F.cross_entropy(logits.float().reshape(-1, BV), tt.reshape(-1))
            loss.backward()
            topt.step()

        step(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1000
        print(f"torch bf16-autocast+SDPA: {ms:7.0f} ms/step | {N/(ms/1000):7.0f} tok/s "
              f"| mem {used_mb()} MiB", flush=True)
        return ("torch bf16-autocast", ms, N / (ms / 1000), used_mb())

    # ---- Axis engine ----
    _compile_runtime()
    from axis.compile import CompiledTransformer
    w = _bench_weights(np.random.default_rng(2))
    cfg = dict(vocab_size=BV, dim=BD, n_layers=BL, n_heads=BH, n_kv_heads=BKV,
               mlp_hidden=BMLP, tie_embeddings=True)
    ct = CompiledTransformer("/root/libaxeng.so", cfg, w, BB, BT,
                             tf32=True, recompute_attn=True, dtype=mode)
    losses = [round(ct.step(inp, tgt), 3) for _ in range(2)]
    ct.capture()
    ct.replay_step(inp, tgt)
    t0 = time.perf_counter()
    for _ in range(5):
        ct.replay_step(inp, tgt)
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"Axis {mode} (CUDA graph): {ms:7.0f} ms/step | {N/(ms/1000):7.0f} tok/s "
          f"| mem {used_mb()} MiB | first losses {losses}", flush=True)
    return (f"axis {mode}", ms, N / (ms / 1000), used_mb())


@app.local_entrypoint()
def main():
    parity.remote()
    rows = [bench.remote(m) for m in ("torch", "fp32", "bf16")]
    print("\n== 1B-class, seq 2048, batch 4 ==")
    for name, ms, tps, mem in rows:
        print(f"{name:22s} {ms:7.0f} ms/step  {tps:8,.0f} tok/s  {mem:6d} MiB")
