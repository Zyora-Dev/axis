"""Phase 3: (1) multi-block compiled-step parity vs the eager oracle,
(2) HONEST benchmark at 1B-class scale — compiled engine vs PyTorch, same
model/shape/GPU (PyTorch first, then freed, then engine).

    modal run engine/modal_compile_test.py
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
app = modal.App("axis-engine-compile")


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

    # ================= 1) multi-block parity (tied embeddings) ==============
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
    ct = CompiledTransformer("/root/libaxeng.so", cfg, weights, B, T, tf32=False)
    got_loss = ct.step(inp, tgt, t=1)
    got_grads = ct.grads()
    worst = ("", 0.0)
    for nm, g in ref_grads.items():
        rel = np.abs(got_grads[nm] - g).max() / (np.abs(g).max() + 1e-9)
        if rel > worst[1]:
            worst = (nm, rel)
    print(f"3-block parity: loss Δ {abs(ref_loss-got_loss):.2e} | worst grad rel {worst[1]:.2e} ({worst[0]})", flush=True)
    ok = abs(ref_loss - got_loss) < 1e-4 and worst[1] < 1e-3
    print(f"MULTI-BLOCK PARITY: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        return

    # ================= 2) 1B-class benchmark vs PyTorch =====================
    Vb, Bb, Tb, Db, Hb, KVb, MLPb, Lb = 32000, 8, 512, 1536, 24, 8, 4096, 48
    Nb = Bb * Tb
    DHb = Db // Hb
    print(f"\n1B-class: dim {Db}, {Lb} layers, {Hb}h/{KVb}kv, mlp {MLPb}, seq {Tb}, batch {Bb}", flush=True)
    rng = np.random.default_rng(1)
    toks_b = rng.integers(0, Vb, size=(Bb, Tb + 1)).astype(np.int64)
    inp_b, tgt_b = toks_b[:, :-1], toks_b[:, 1:]

    # ---- PyTorch reference (plain torch Llama-class, TF32, AdamW) ----
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
            return s.norm(x) @ s.emb.weight.T   # tied

    tm = TModel().cuda().float()
    n_params = sum(p.numel() for p in tm.parameters())
    print(f"params: {n_params/1e9:.2f}B", flush=True)
    topt = torch.optim.AdamW(tm.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    ti = torch.tensor(inp_b).cuda(); tt = torch.tensor(tgt_b).cuda()

    def torch_step():
        topt.zero_grad(set_to_none=True)
        logits = tm(ti)
        loss = F.cross_entropy(logits.reshape(-1, Vb), tt.reshape(-1))
        loss.backward()
        topt.step()

    torch_step(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        torch_step()
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"PyTorch : {torch_ms:7.0f} ms/step | {Nb/(torch_ms/1000):7.0f} tok/s", flush=True)
    del tm, topt, ti, tt
    torch.cuda.empty_cache()

    # ---- Axis compiled engine (TF32, graph-captured) ----
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
    ct = CompiledTransformer("/root/libaxeng.so", cfg_b, w1b, Bb, Tb, tf32=True)

    losses = [round(ct.step(inp_b, tgt_b, t=k + 1), 3) for k in range(2)]  # warm + sanity
    t0 = time.perf_counter()
    for k in range(5):
        ct.step(inp_b, tgt_b, t=k + 3)
    plan_ms = (time.perf_counter() - t0) / 5 * 1000

    ct.capture(t=10)
    ct.replay_step(inp_b, tgt_b)
    t0 = time.perf_counter()
    for _ in range(5):
        ct.replay_step(inp_b, tgt_b)
    graph_ms = (time.perf_counter() - t0) / 5 * 1000

    print(f"Axis eng: {plan_ms:7.0f} ms/step | {Nb/(plan_ms/1000):7.0f} tok/s (plan)", flush=True)
    print(f"Axis eng: {graph_ms:7.0f} ms/step | {Nb/(graph_ms/1000):7.0f} tok/s (CUDA graph)", flush=True)
    print(f"first losses (sanity, should fall): {losses}", flush=True)
    print(f"\n>>> Axis engine vs PyTorch at 1B-class: {torch_ms/graph_ms:.2f}x "
          f"({Nb/(graph_ms/1000):,.0f} vs {Nb/(torch_ms/1000):,.0f} tok/s)", flush=True)


@app.local_entrypoint()
def main():
    test.remote()
