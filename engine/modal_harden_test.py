"""Engine hardening pass:
1) SHAPE FUZZING — 16 configs (curated edges + seeded random) vs the eager
   oracle: DH=128 flash branch, DH=16, L=1/B=1/ragged-T, untied embeddings,
   MHA, MQA, odd vocab, LoRA with random rank/targets. Each config gated
   fp32 (bit-tight) AND bf16 (relaxed).
2) SOAK — ~150M model, 500 CUDA-graph replays on repeating data: loss must
   fall (memorization), no NaN/inf ever, plan-path continues consistently.
3) LOSS-CURVE EQUIVALENCE — 30 steps eager fp32 vs compiled fp32 vs
   compiled bf16 on identical data: fp32 must track the oracle ~exactly,
   bf16 within a small drift.

    modal run engine/modal_harden_test.py
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
app = modal.App("axis-engine-harden")


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


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def fuzz():
    import sys
    import numpy as np
    sys.path.insert(0, "/root")
    _compile_runtime()
    import axis
    from axis import nn, lora
    from axis.tensor import Tensor
    from axis.compile import compile_model

    # (V, B, T, D, H, KV, MLP, L, tied, lora_r, targets, tile)
    LORA_T = ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")
    cases = [
        ("DH=128 flash branch", 300, 2, 96, 1024, 8, 4, 512, 2, True, 0, (), 32),
        ("DH=16 tiny heads    ", 200, 2, 40, 64, 4, 2, 128, 2, True, 0, (), 8),
        ("L=1 B=1 T=17 ragged ", 150, 1, 17, 128, 4, 2, 256, 1, True, 0, (), 5),
        ("untied embeddings   ", 250, 2, 32, 128, 4, 2, 256, 2, False, 0, (), 8),
        ("MHA KV=H odd V=97   ", 97, 2, 48, 128, 4, 4, 256, 2, True, 0, (), 16),
        ("MQA KV=1            ", 300, 2, 64, 192, 6, 1, 384, 2, True, 0, (), 16),
        ("LoRA r=2 qv only    ", 300, 2, 32, 128, 4, 2, 256, 2, True, 2, ("q_proj", "v_proj"), 8),
        ("LoRA r=8 all+untied ", 300, 2, 32, 128, 4, 2, 256, 2, False, 8, LORA_T, 8),
    ]
    rng = np.random.default_rng(42)
    for ci in range(8):                     # random configs
        H = int(rng.choice([2, 4, 6, 8]))
        KV = int(rng.choice([g for g in range(1, H + 1) if H % g == 0]))
        DH = int(rng.choice([16, 32, 48, 64]))
        L = int(rng.integers(1, 4))
        T = int(rng.integers(9, 80))
        B = int(rng.integers(1, 4))
        V = int(rng.integers(80, 600))
        MLP = int(rng.choice([96, 160, 288]))
        tied = bool(rng.integers(0, 2))
        lr_ = int(rng.choice([0, 0, 4]))    # 1/3 of randoms are LoRA
        tg = tuple(rng.choice(LORA_T, size=3, replace=False)) if lr_ else ()
        tile = int(rng.choice([5, 8, 16, 64]))
        cases.append((f"rand{ci} D{H*DH} H{H}/{KV} T{T} B{B} L{L}"
                      f"{' lora' if lr_ else ''}", V, B, T, H * DH, H, KV,
                      MLP, L, tied, lr_, tg, tile))

    ok_all = True
    for (label, V, B, T, D, H, KV, MLP, L, tied, lr_, tg, tile) in cases:
        def build():
            axis.manual_seed(7)
            m = nn.Transformer(vocab_size=V, dim=D, n_layers=L, n_heads=H,
                               n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=T,
                               tie_embeddings=tied)
            if lr_:
                lora.apply_lora(m, rank=lr_, alpha=2 * lr_, target_modules=tg)
            return m
        drng = np.random.default_rng(1)
        toks = drng.integers(0, V, size=(B, T + 1)).astype(np.int64)
        inp, tgt = toks[:, :-1], toks[:, 1:]
        model = build()
        lt = model.loss(Tensor(inp), Tensor(tgt))
        rl = float(lt.data)
        lt.backward()
        rg = {n: p.grad.copy() for n, p in model.named_parameters()
              if p.requires_grad}

        row = f"{label}: "
        for dtype, ltol, gtol in (("fp32", 1e-4, 1e-3), ("bf16", 5e-2, 1.5e-1)):
            ct = compile_model(build(), B, T, lib_path="/root/libaxeng.so",
                               tf32=False, dtype=dtype, attn_tile=tile)
            gl = ct.step(inp, tgt, t=1)
            gg = ct.grads()
            worst = max(np.abs(gg[nm] - g).max() / (np.abs(g).max() + 1e-9)
                        for nm, g in rg.items())
            okx = abs(rl - gl) < ltol and worst < gtol and np.isfinite(gl)
            ok_all &= okx
            row += (f"{dtype}[{ct.attn_impl[0]}] Δ{abs(rl-gl):.1e} "
                    f"g{worst:.1e} {'✓' if okx else 'FAIL'}  ")
        print(row, flush=True)
    print(f"\nFUZZ: {'PASS' if ok_all else 'FAIL'}", flush=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def soak():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    _compile_runtime()
    from axis.compile import CompiledTransformer

    V, B, T, D, H, KV, MLP, L = 8000, 8, 512, 768, 12, 4, 2048, 12
    DH = D // H
    shapes = {"embed.weight": (V, D), "norm.weight": (D,)}
    for i in range(L):
        p = f"blocks.{i}."
        shapes.update({p + "attn_norm.weight": (D,), p + "mlp_norm.weight": (D,),
                       p + "attn.q_proj.weight": (D, H * DH),
                       p + "attn.k_proj.weight": (D, KV * DH),
                       p + "attn.v_proj.weight": (D, KV * DH),
                       p + "attn.o_proj.weight": (H * DH, D),
                       p + "mlp.gate_proj.weight": (D, MLP),
                       p + "mlp.up_proj.weight": (D, MLP),
                       p + "mlp.down_proj.weight": (MLP, D)})
    r2 = np.random.default_rng(2)
    w = {nm: (np.ones(sh, dtype=np.float32) if "norm" in nm else
              (r2.standard_normal(sh) * 0.02).astype(np.float32))
         for nm, sh in shapes.items()}
    n_par = sum(int(np.prod(s)) for s in shapes.values())
    print(f"soak model: {n_par/1e6:.0f}M params", flush=True)
    cfg = dict(vocab_size=V, dim=D, n_layers=L, n_heads=H, n_kv_heads=KV,
               mlp_hidden=MLP, tie_embeddings=True)
    ct = CompiledTransformer("/root/libaxeng.so", cfg, w, B, T,
                             lr=3e-4, tf32=True, dtype="bf16")
    rng = np.random.default_rng(3)
    batches = [rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
               for _ in range(8)]                       # repeating -> memorizable
    ct.step(batches[0][:, :-1], batches[0][:, 1:])      # warm (also builds bufs)
    ct.capture()
    losses = []
    t0 = time.perf_counter()
    STEPS = 500
    for s in range(STEPS):
        b = batches[s % 8]
        ls = ct.replay_step(b[:, :-1], b[:, 1:])
        if not np.isfinite(ls):
            print(f"NaN/inf at step {s}: {ls} — FAIL", flush=True)
            return
        losses.append(ls)
        if s % 100 == 0:
            print(f"step {s:4d}: loss {ls:.4f}", flush=True)
    dt = time.perf_counter() - t0
    # continue on the plan path (non-graph) — must be consistent
    ls_plan = ct.step(batches[0][:, :-1], batches[0][:, 1:])
    first, last = np.mean(losses[:8]), np.mean(losses[-8:])
    ok = (last < first - 1.0 and np.isfinite(ls_plan)
          and abs(ls_plan - losses[-8]) < 1.0)
    print(f"500 graph replays in {dt:.0f}s ({STEPS*B*T/dt:,.0f} tok/s) | "
          f"loss {first:.3f} -> {last:.3f} | plan-path continue {ls_plan:.3f}",
          flush=True)
    print(f"SOAK: {'PASS' if ok else 'FAIL'}", flush=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def curve():
    import sys
    import numpy as np
    sys.path.insert(0, "/root")
    _compile_runtime()
    import axis
    from axis import nn, optim
    from axis.tensor import Tensor
    from axis.compile import compile_model

    V, B, T, D, H, KV, MLP, L = 500, 2, 32, 128, 4, 2, 256, 3
    STEPS, LR = 30, 1e-3
    rng = np.random.default_rng(9)
    batches = [rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
               for _ in range(STEPS)]

    def build():
        axis.manual_seed(0)
        return nn.Transformer(vocab_size=V, dim=D, n_layers=L, n_heads=H,
                              n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=T,
                              tie_embeddings=True)

    # eager oracle training loop
    model = build()
    opt = optim.AdamW(model.parameters(), lr=LR)
    ref = []
    for b in batches:
        opt.zero_grad()
        lt = model.loss(Tensor(b[:, :-1]), Tensor(b[:, 1:]))
        lt.backward()
        opt.step()
        ref.append(float(lt.data))

    def run(dtype, impl):
        ct = compile_model(build(), B, T, lib_path="/root/libaxeng.so",
                           lr=LR, tf32=False, dtype=dtype, attn_impl=impl)
        return [ct.step(b[:, :-1], b[:, 1:]) for b in batches]

    c32 = run("fp32", "auto")
    cbf = run("bf16", "flash")
    d32 = max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(ref, c32))
    dbf = max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(ref, cbf))
    print("step   eager    fp32     bf16", flush=True)
    for i in (0, 9, 19, 29):
        print(f"{i+1:4d} {ref[i]:8.4f} {c32[i]:8.4f} {cbf[i]:8.4f}", flush=True)
    ok = d32 < 1e-3 and dbf < 3e-2
    print(f"max rel drift vs eager oracle: fp32 {d32:.2e} | bf16 {dbf:.2e}",
          flush=True)
    print(f"CURVE: {'PASS' if ok else 'FAIL'}", flush=True)


@app.local_entrypoint()
def main():
    fuzz.remote()
    soak.remote()
    curve.remote()
