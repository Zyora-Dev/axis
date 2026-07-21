"""Validate the unknowns-fixes:
1. recompute_attn parity — grads still exactly match the eager oracle
2. device-side AdamW bias correction — graph replays train correctly
3. 1.26B at seq 2048 (real training shape) — memory + tok/s vs PyTorch

    modal run engine/modal_longseq_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "torch")
    .add_local_file(str(REPO / "axis" / "_csrc" / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-engine-longseq")


@app.function(image=image, gpu="A100-80GB", timeout=3600)
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
    from axis.compile import CompiledTransformer

    # ---- 1) recompute-mode parity vs eager oracle ----
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
    ct = CompiledTransformer("/root/libaxeng.so", cfg, weights, B, T,
                             tf32=False, recompute_attn=True)
    got_loss = ct.step(inp, tgt, t=1)
    got = ct.grads()
    worst = max((np.abs(got[nm] - g).max() / (np.abs(g).max() + 1e-9), nm)
                for nm, g in ref_grads.items())
    print(f"1) recompute parity: loss Δ {abs(ref_loss-got_loss):.2e} | worst grad rel {worst[0]:.2e}"
          f" -> {'PASS' if worst[0] < 1e-3 else 'FAIL'}", flush=True)

    # ---- 2) device-t AdamW: graph replays must train with exact bias corr ----
    ct2 = CompiledTransformer("/root/libaxeng.so", cfg, weights, B, T,
                              tf32=False, recompute_attn=True, lr=3e-3)
    ct2.capture()             # device-side t (TICK in graph)
    losses = [ct2.replay_step(inp, tgt) for _ in range(30)]
    print(f"2) graph-replay training (device-t AdamW): loss {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"-> {'PASS' if losses[-1] < losses[0] * 0.7 else 'FAIL'}", flush=True)

    # ---- 3) 1.26B @ seq 2048 (real shape) vs PyTorch ----
    Vb, Bb, Tb, Db, Hb, KVb, MLPb, Lb = 32000, 4, 2048, 1536, 24, 8, 4096, 48
    Nb = Bb * Tb
    DHb = Db // Hb
    print(f"\n3) 1.26B @ seq {Tb}, batch {Bb} (the real-training shape)", flush=True)
    rngb = np.random.default_rng(1)
    toks_b = rngb.integers(0, Vb, size=(Bb, Tb + 1)).astype(np.int64)
    inp_b, tgt_b = toks_b[:, :-1], toks_b[:, 1:]

    # PyTorch reference first (then freed)
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = True

    class TRms(tnn.Module):
        def __init__(s, d):
            super().__init__(); s.w = tnn.Parameter(torch.ones(d))
        def forward(s, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * s.w

    class TBlock(tnn.Module):
        def __init__(s):
            super().__init__()
            s.n1, s.n2 = TRms(Db), TRms(Db)
            s.q = tnn.Linear(Db, Hb * DHb, bias=False)
            s.k = tnn.Linear(Db, KVb * DHb, bias=False)
            s.v = tnn.Linear(Db, KVb * DHb, bias=False)
            s.o = tnn.Linear(Hb * DHb, Db, bias=False)
            s.g = tnn.Linear(Db, MLPb, bias=False)
            s.u = tnn.Linear(Db, MLPb, bias=False)
            s.dn = tnn.Linear(MLPb, Db, bias=False)
        def forward(s, x, cos, sin):
            Bx, Tx, _ = x.shape
            y = s.n1(x)
            q = s.q(y).view(Bx, Tx, Hb, DHb).transpose(1, 2)
            k = s.k(y).view(Bx, Tx, KVb, DHb).transpose(1, 2)
            v = s.v(y).view(Bx, Tx, KVb, DHb).transpose(1, 2)
            def rope(t):
                h = DHb // 2
                t1, t2 = t[..., :h], t[..., h:]
                return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], -1)
            q, k = rope(q), rope(k)
            k = k.repeat_interleave(Hb // KVb, dim=1)
            v = v.repeat_interleave(Hb // KVb, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(Bx, Tx, Hb * DHb)
            x = x + s.o(a)
            z = s.n2(x)
            return x + s.dn(F.silu(s.g(z)) * s.u(z))

    class TModel(tnn.Module):
        def __init__(s):
            super().__init__()
            s.emb = tnn.Embedding(Vb, Db)
            s.blocks = tnn.ModuleList([TBlock() for _ in range(Lb)])
            s.norm = TRms(Db)
            half = DHb // 2
            fr = 1.0 / (10000.0 ** (torch.arange(half) / half))
            ang = torch.outer(torch.arange(Tb), fr)
            s.register_buffer("cos", ang.cos()[None, None])
            s.register_buffer("sin", ang.sin()[None, None])
        def forward(s, ids):
            x = s.emb(ids)
            for b in s.blocks:
                x = b(x, s.cos, s.sin)
            return s.norm(x) @ s.emb.weight.T

    torch_ms = None
    try:
        tm = TModel().cuda().float()
        topt = torch.optim.AdamW(tm.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        ti = torch.tensor(inp_b).cuda(); tt = torch.tensor(tgt_b).cuda()
        def tstep():
            topt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(tm(ti).reshape(-1, Vb), tt.reshape(-1))
            loss.backward(); topt.step()
        tstep(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(3):
            tstep()
        torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - t0) / 3 * 1000
        tmem = torch.cuda.max_memory_allocated() / 1e9
        print(f"   PyTorch : {torch_ms:7.0f} ms | {Nb/(torch_ms/1000):7.0f} tok/s | peak {tmem:.1f}G", flush=True)
        del tm, topt, ti, tt
    except torch.cuda.OutOfMemoryError:
        print("   PyTorch : OOM at this shape (eager, no grad ckpt)", flush=True)
    torch.cuda.empty_cache()

    # Axis compiled engine, recompute mode
    w1b = {}
    r2 = np.random.default_rng(2)
    shapes = {"embed.weight": (Vb, Db), "norm.weight": (Db,)}
    for i in range(Lb):
        p = f"blocks.{i}."
        shapes.update({p + "attn_norm.weight": (Db,), p + "mlp_norm.weight": (Db,),
                       p + "attn.q_proj.weight": (Db, Hb * DHb),
                       p + "attn.k_proj.weight": (Db, KVb * DHb),
                       p + "attn.v_proj.weight": (Db, KVb * DHb),
                       p + "attn.o_proj.weight": (Hb * DHb, Db),
                       p + "mlp.gate_proj.weight": (Db, MLPb),
                       p + "mlp.up_proj.weight": (Db, MLPb),
                       p + "mlp.down_proj.weight": (MLPb, Db)})
    for nm, sh in shapes.items():
        w1b[nm] = np.ones(sh, dtype=np.float32) if "norm" in nm else \
            (r2.standard_normal(sh) * 0.02).astype(np.float32)
    cfg_b = dict(vocab_size=Vb, dim=Db, n_layers=Lb, n_heads=Hb, n_kv_heads=KVb,
                 mlp_hidden=MLPb, tie_embeddings=True)
    ct3 = CompiledTransformer("/root/libaxeng.so", cfg_b, w1b, Bb, Tb,
                              tf32=True, recompute_attn=True)
    l0 = ct3.step(inp_b, tgt_b)   # warm (device-t)
    ct3.capture()
    ct3.replay_step(inp_b, tgt_b)
    t0 = time.perf_counter()
    for _ in range(3):
        ct3.replay_step(inp_b, tgt_b)
    graph_ms = (time.perf_counter() - t0) / 3 * 1000
    print(f"   Axis eng: {graph_ms:7.0f} ms | {Nb/(graph_ms/1000):7.0f} tok/s (CUDA graph, recompute)", flush=True)
    if torch_ms:
        print(f"\n>>> 1.26B @ seq2048: Axis vs PyTorch = {torch_ms/graph_ms:.2f}x", flush=True)


@app.local_entrypoint()
def main():
    test.remote()
