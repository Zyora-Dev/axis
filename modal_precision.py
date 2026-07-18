"""Measure TF32 tensor-core speedup (safe: fp32 dtype kept, ~fp32 precision)
vs plain fp32 cuBLAS on A100. TF32 is what PyTorch enables by default on
Ampere. Two fresh containers (env var read at cupy import).

    modal run modal_precision.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent


def _image():
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
        .pip_install("numpy>=1.24", "cupy-cuda12x")
        .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
    )


app = modal.App("axis-precision")
DIM, LAYERS, HEADS, KV, MLP = 512, 6, 8, 4, 1024
BATCH, SEQ, STEPS = 8, 128, 25


@app.function(image=_image(), gpu="A100", timeout=1800)
def bench(tf32: bool):
    import os
    os.environ["CUPY_TF32"] = "1" if tf32 else "0"  # must precede cupy import
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import axis
    from axis import nn, optim, backend, ByteTokenizer, LMDataset, DataLoader

    ids = ByteTokenizer().encode("Axis trains fast on GPU with tensor cores. " * 500)
    ds = LMDataset(ids, seq_len=SEQ)
    backend.set_device("gpu")
    axis.manual_seed(0)
    m = nn.Transformer(vocab_size=256, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                       n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=256).to_gpu()
    opt = optim.AdamW(m.parameters(), lr=3e-4)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, seed=0)
    import cupy as cp
    for inp, tgt in dl:  # warmup
        l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad(); break
    losses, ms = [], []
    step = 0
    for inp, tgt in dl:
        t0 = time.perf_counter()
        l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()
        cp.cuda.Stream.null.synchronize()
        ms.append((time.perf_counter() - t0) * 1000); losses.append(float(l.data))
        step += 1
        if step >= STEPS:
            break
    med = float(np.median(ms))
    label = "TF32 tensor cores" if tf32 else "fp32 (baseline)"
    print(f"[{label}] {med:.0f} ms/step | {BATCH*SEQ/(med/1000):.0f} tok/s | "
          f"loss {losses[0]:.3f}->{losses[-1]:.3f}", flush=True)
    return label, med, BATCH * SEQ / (med / 1000), losses[0], losses[-1]


@app.local_entrypoint()
def main():
    r0 = bench.remote(False)
    r1 = bench.remote(True)
    print("\n=== PRECISION COMPARISON (A100, dim512/6L) ===")
    for label, med, toks, l0, l1 in (r0, r1):
        print(f"  {label:20s}: {med:5.0f} ms | {toks:6.0f} tok/s | loss {l0:.3f}->{l1:.3f}")
    print(f"  >>> TF32 speedup: {r0[1]/r1[1]:.2f}x  ({r0[2]:.0f} -> {r1[2]:.0f} tok/s)")
