"""Batch-size sweep on A100 — tok/s vs batch (safe: no precision change, just
better GPU utilization). Finds the throughput sweet spot.

    modal run modal_batch_sweep.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-batch-sweep")
DIM, LAYERS, HEADS, KV, MLP, SEQ = 512, 6, 8, 4, 1024, 128


@app.function(image=image, gpu="A100", timeout=1800)
def sweep():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import cupy as cp
    import axis
    from axis import nn, optim, backend, ByteTokenizer, LMDataset, DataLoader

    ids = ByteTokenizer().encode("Axis scales throughput with batch size on GPU. " * 2000)
    ds = LMDataset(ids, seq_len=SEQ)
    backend.set_device("gpu")

    print("batch   ms/step   tok/s", flush=True)
    best = (0, 0)
    for batch in (8, 16, 32, 64, 96):
        axis.manual_seed(0)
        m = nn.Transformer(vocab_size=256, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                           n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=256).to_gpu()
        opt = optim.AdamW(m.parameters(), lr=3e-4)
        dl = DataLoader(ds, batch_size=batch, shuffle=True, seed=0)
        ms = []
        step = 0
        for inp, tgt in dl:
            if step == 0:  # warmup
                l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()
                step = 1
                continue
            t0 = time.perf_counter()
            l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()
            cp.cuda.Stream.null.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
            step += 1
            if step >= 12:
                break
        med = float(np.median(ms))
        toks = batch * SEQ / (med / 1000)
        print(f"{batch:5d}   {med:7.0f}   {toks:6.0f}", flush=True)
        if toks > best[1]:
            best = (batch, toks)
    print(f"\n>>> best: batch {best[0]} = {best[1]:.0f} tok/s", flush=True)


@app.local_entrypoint()
def main():
    sweep.remote()
