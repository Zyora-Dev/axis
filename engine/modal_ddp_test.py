"""Multi-GPU data parallel (#5) validation on 2x A100:
1) PARITY — 2 ranks, same init, DIFFERENT micro-batches, grads averaged by
   in-plan NCCL allreduce: rank0's post-allreduce grads and the mean loss
   must EXACTLY match a single-GPU step on the concatenated batch (fp32
   bit-tight; the math is identical because per-rank CE means average to
   the global mean at equal batch sizes). 5-step weight-trajectory check.
2) SCALING — 1B-class bf16+flash, batch 4/rank: tokens/s on 2 GPUs vs 1,
   weak-scaling efficiency.

    modal run engine/modal_ddp_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "nvidia-nccl-cu12")
    .add_local_file(str(REPO / "axis" / "_csrc" / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-engine-ddp")

# 1B-class shape
BV, BT, BD, BH, BKV, BMLP, BL = 32000, 2048, 1536, 24, 8, 4096, 48


def _nccl_dir():
    import nvidia.nccl
    return list(nvidia.nccl.__path__)[0]   # namespace pkg: __file__ is None


def _compile_runtime():
    import subprocess
    nd = _nccl_dir()
    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "--shared", "-Xcompiler", "-fPIC",
         "-DAXIS_NCCL", f"-I{nd}/include", "/root/runtime.cu",
         f"-L{nd}/lib", "-lnccl", "-lcublas", "-o", "/root/libaxeng.so"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE FAILED:\n", r.stderr[-3000:], flush=True)
        raise SystemExit(1)
    print("nvcc (+NCCL): OK", flush=True)


def _load_nccl():
    """Preload libnccl so libaxeng.so resolves it regardless of LD_LIBRARY_PATH."""
    import ctypes
    ctypes.CDLL(f"{_nccl_dir()}/lib/libnccl.so.2", mode=ctypes.RTLD_GLOBAL)


def _worker(rank, world, uid_q, res_q, mode, payload):
    import os, sys, time
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    sys.path.insert(0, "/root")
    import numpy as np
    _load_nccl()
    from axis.compile import CompiledTransformer

    cfg, weights, B, T, kw = payload
    ct = CompiledTransformer("/root/libaxeng.so", cfg, weights, B, T,
                             grad_sync=True, **kw)
    if rank == 0:
        uid = ct.eng.nccl_id()
        for _ in range(world - 1):
            uid_q.put(uid)
    else:
        uid = uid_q.get(timeout=120)
    ct.eng.nccl_init(rank, world, uid)

    rng = np.random.default_rng(100 + rank)     # DIFFERENT data per rank
    if mode == "parity":
        losses, gsnap, wsnap = [], None, None
        for s in range(5):
            toks = rng.integers(0, cfg["vocab_size"], size=(B, T + 1)).astype(np.int64)
            ls = ct.step(toks[:, :-1], toks[:, 1:], t=s + 1)
            losses.append(ls)
            if s == 0 and rank == 0:
                gsnap = ct.grads()
        if rank == 0:
            wsnap = ct.get_weights()
        res_q.put((rank, losses, gsnap, wsnap, None))
    else:                                       # bench
        toks = rng.integers(0, cfg["vocab_size"], size=(B, T + 1)).astype(np.int64)
        inp, tgt = toks[:, :-1], toks[:, 1:]
        ct.step(inp, tgt); ct.step(inp, tgt)    # warm
        t0 = time.perf_counter()
        for _ in range(5):
            ct.step(inp, tgt)
        ms = (time.perf_counter() - t0) / 5 * 1000
        res_q.put((rank, None, None, None, ms))


def _spawn(world, mode, payload):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    uid_q, res_q = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, uid_q, res_q, mode, payload))
             for r in range(world)]
    for p in procs:
        p.start()
    out = [res_q.get(timeout=1200) for _ in range(world)]
    for p in procs:
        p.join(timeout=60)
    return sorted(out)


def _weights_1b(seed=2):
    import numpy as np
    DH = BD // BH
    shapes = {"embed.weight": (BV, BD), "norm.weight": (BD,)}
    for i in range(BL):
        p = f"blocks.{i}."
        shapes.update({p + "attn_norm.weight": (BD,), p + "mlp_norm.weight": (BD,),
                       p + "attn.q_proj.weight": (BD, BH * DH),
                       p + "attn.k_proj.weight": (BD, BKV * DH),
                       p + "attn.v_proj.weight": (BD, BKV * DH),
                       p + "attn.o_proj.weight": (BH * DH, BD),
                       p + "mlp.gate_proj.weight": (BD, BMLP),
                       p + "mlp.up_proj.weight": (BD, BMLP),
                       p + "mlp.down_proj.weight": (BMLP, BD)})
    r = np.random.default_rng(seed)
    return {nm: (np.ones(sh, dtype=np.float32) if "norm" in nm else
                 (r.standard_normal(sh) * 0.02).astype(np.float32))
            for nm, sh in shapes.items()}


@app.function(image=image, gpu="A100-80GB:2", timeout=3600, memory=65536)
def parity():
    import sys
    import numpy as np
    sys.path.insert(0, "/root")
    _compile_runtime()

    V, B, T, D, H, KV, MLP, L = 500, 2, 32, 128, 4, 2, 256, 3
    DH = D // H
    r = np.random.default_rng(5)
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
    w = {nm: (np.ones(sh, dtype=np.float32) if "norm" in nm else
              (r.standard_normal(sh) * 0.02).astype(np.float32))
         for nm, sh in shapes.items()}
    cfg = dict(vocab_size=V, dim=D, n_layers=L, n_heads=H, n_kv_heads=KV,
               mlp_hidden=MLP, tie_embeddings=True)
    kw = dict(tf32=False, dtype="fp32")

    out = _spawn(2, "parity", (cfg, {k: v.copy() for k, v in w.items()}, B, T, kw))
    (r0, l0, g0, w0, _), (r1, l1, _, _, _) = out

    # single-GPU reference on the CONCATENATED batch (same per-rank data streams)
    from axis.compile import CompiledTransformer
    _load_nccl()
    rngs = [np.random.default_rng(100 + k) for k in range(2)]
    ct = CompiledTransformer("/root/libaxeng.so", cfg, w, 2 * B, T, **kw)
    ref_losses, ref_g = [], None
    for s in range(5):
        parts = [rg.integers(0, V, size=(B, T + 1)).astype(np.int64) for rg in rngs]
        toks = np.concatenate(parts, axis=0)
        rl = ct.step(toks[:, :-1], toks[:, 1:], t=s + 1)
        ref_losses.append(rl)
        if s == 0:
            ref_g = ct.grads()
    ref_w = ct.get_weights()

    ok = True
    for s in range(5):
        dd = abs((l0[s] + l1[s]) / 2 - ref_losses[s])
        ok &= dd < 1e-4
        print(f"step {s+1}: rank-mean loss {(l0[s]+l1[s])/2:.6f} vs "
              f"single-GPU {ref_losses[s]:.6f} | Δ {dd:.2e}", flush=True)
    worst = max(np.abs(g0[nm] - g).max() / (np.abs(g).max() + 1e-9)
                for nm, g in ref_g.items())
    wworst = max(np.abs(w0[nm] - wv).max() for nm, wv in ref_w.items())
    ok &= worst < 1e-3 and wworst < 1e-4
    print(f"post-allreduce grads vs full-batch: worst rel {worst:.2e}", flush=True)
    print(f"weights after 5 steps: max |Δ| {wworst:.2e}", flush=True)
    print(f"DDP PARITY: {'PASS' if ok else 'FAIL'}", flush=True)


@app.function(image=image, gpu="A100-80GB:2", timeout=3600, memory=65536,
              single_use_containers=True)
def bench_ddp():
    import sys
    sys.path.insert(0, "/root")
    _compile_runtime()
    cfg = dict(vocab_size=BV, dim=BD, n_layers=BL, n_heads=BH, n_kv_heads=BKV,
               mlp_hidden=BMLP, tie_embeddings=True)
    kw = dict(tf32=True, dtype="bf16")
    out = _spawn(2, "bench", (cfg, _weights_1b(), 4, BT, kw))
    ms = max(o[4] for o in out)
    N = 2 * 4 * BT
    print(f"2x A100 DDP: {ms:7.0f} ms/step | {N/(ms/1000):7.0f} tok/s total",
          flush=True)
    return ms, N / (ms / 1000)


@app.function(image=image, gpu="A100-80GB", timeout=3600, memory=65536,
              single_use_containers=True)
def bench_single():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    _compile_runtime()
    _load_nccl()
    from axis.compile import CompiledTransformer
    cfg = dict(vocab_size=BV, dim=BD, n_layers=BL, n_heads=BH, n_kv_heads=BKV,
               mlp_hidden=BMLP, tie_embeddings=True)
    ct = CompiledTransformer("/root/libaxeng.so", cfg, _weights_1b(), 4, BT,
                             tf32=True, dtype="bf16")
    rng = np.random.default_rng(100)
    toks = rng.integers(0, BV, size=(4, BT + 1)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]
    ct.step(inp, tgt); ct.step(inp, tgt)
    t0 = time.perf_counter()
    for _ in range(5):
        ct.step(inp, tgt)
    ms = (time.perf_counter() - t0) / 5 * 1000
    N = 4 * BT
    print(f"1x A100 (plan): {ms:7.0f} ms/step | {N/(ms/1000):7.0f} tok/s",
          flush=True)
    return ms, N / (ms / 1000)


@app.local_entrypoint()
def main():
    parity.remote()
    ms1, tps1 = bench_single.remote()
    ms2, tps2 = bench_ddp.remote()
    print(f"\n== 1B-class bf16+flash, batch 4/GPU, seq {BT} ==")
    print(f"1x A100: {ms1:7.0f} ms/step  {tps1:8,.0f} tok/s")
    print(f"2x A100: {ms2:7.0f} ms/step  {tps2:8,.0f} tok/s "
          f"| weak-scaling efficiency {tps2/(2*tps1)*100:.0f}%")
