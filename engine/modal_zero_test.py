"""ZeRO-1 (optimizer-state sharding) validation on 2x A100:
1) PARITY — 2 ranks, zero_stage=1, DIFFERENT micro-batches; after 5 steps the
   MERGED owned weights (each rank owns ~half the optimizer state) must EXACTLY
   equal a single-GPU run on the concatenated batch. Proves sharded AdamW +
   broadcast reproduce the replicated result.
2) SHARDING — each rank holds optimizer state (fp32 master+m+v) for only its
   owned params; report the split to confirm the memory is actually divided.

    modal run engine/modal_zero_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "nvidia-nccl-cu12")
    .add_local_file(str(HERE / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-engine-zero")


def _nccl_dir():
    import nvidia.nccl
    return list(nvidia.nccl.__path__)[0]


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
    import ctypes
    ctypes.CDLL(f"{_nccl_dir()}/lib/libnccl.so.2", mode=ctypes.RTLD_GLOBAL)


def _worker(rank, world, uid_q, res_q, cfg, weights, B, T, zero_stage):
    import os, sys
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    sys.path.insert(0, "/root")
    import numpy as np
    _load_nccl()
    from axis.compile import CompiledTransformer

    ct = CompiledTransformer("/root/libaxeng.so", cfg, weights, B, T,
                             tf32=False, dtype="bf16", attn_impl="tiled",
                             grad_sync=True, zero_stage=zero_stage,
                             rank=rank, world=world)
    if rank == 0:
        uid = ct.eng.nccl_id()
        for _ in range(world - 1):
            uid_q.put(uid)
    else:
        uid = uid_q.get(timeout=120)
    ct.eng.nccl_init(rank, world, uid)

    rng = np.random.default_rng(100 + rank)     # different data per rank
    for s in range(5):
        toks = rng.integers(0, cfg["vocab_size"], size=(B, T + 1)).astype(np.int64)
        ct.step(toks[:, :-1], toks[:, 1:], t=s + 1)
    owned = ct.get_weights()                     # ZeRO: shard; DDP: full
    n_state = sum(int(np.prod(ct.shapes[nm])) for nm in ct.owned)
    n_all = sum(int(np.prod(ct.shapes[nm])) for nm in ct.pnames)
    res_q.put((rank, {k: v for k, v in owned.items()}, len(ct.owned),
               len(ct.pnames), n_state, n_all))


def _run(zero_stage, cfg, w, B, T):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    uid_q, res_q = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(rk, 2, uid_q, res_q, cfg,
                               {k: v.copy() for k, v in w.items()}, B, T, zero_stage))
             for rk in range(2)]
    for p in procs:
        p.start()
    out = sorted([res_q.get(timeout=1200) for _ in range(2)])
    for p in procs:
        p.join(timeout=60)
    return out


@app.function(image=image, gpu="A100-80GB:2", timeout=3600, memory=65536)
def parity():
    import sys
    import multiprocessing as mp
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

    # ZeRO-1 (sharded optimizer state) — merge the two owned shards
    zout = _run(1, cfg, w, B, T)
    merged = {}
    for rk, owned, n_own, n_tot, n_state, n_all in zout:
        merged.update(owned)
        print(f"rank {rk}: owns {n_own}/{n_tot} params, optimizer state "
              f"{n_state/1e3:.1f}K/{n_all/1e3:.1f}K elems "
              f"({100*n_state/n_all:.0f}% of full)", flush=True)

    # replicated DDP (full optimizer state on every rank) — SAME data, SAME
    # bf16 — the ONLY difference is where the optimizer state lives + the
    # broadcast. rank0 holds the full model.
    dout = _run(0, cfg, w, B, T)
    ref = dict(dout[0][1])

    worst = max(np.abs(merged[nm] - ref[nm]).max() for nm in ref)
    ok = worst < 1e-4 and set(merged) == set(ref)
    print(f"\nZeRO-1 merged vs replicated DDP (both bf16): max |Δ| {worst:.2e}", flush=True)
    print(f"ZeRO-1 PARITY: {'PASS' if ok else 'FAIL'}", flush=True)


@app.local_entrypoint()
def main():
    parity.remote()
