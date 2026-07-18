"""Scale validation: does Axis train realistically-sized models (100M-350M
params) end-to-end on an A100 — no OOM, no NaN, converging loss, sane tok/s and
memory. Kills the "toy framework" objection before any benchmark.

    modal run modal_scale.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-scale")

# (name, dim, layers, heads, kv, mlp, batch, seq)
CONFIGS = [
    ("~124M (GPT-2 base class)", 768, 12, 12, 4, 2048, 12, 512),
    ("~350M (GPT-2 medium class)", 1024, 24, 16, 4, 2816, 8, 512),
]


@app.function(image=image, gpu="A100", timeout=2400)
def scale():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import cupy as cp
    import axis
    from axis import nn, optim, backend, ByteTokenizer, LMDataset, DataLoader

    ids = ByteTokenizer().encode("Axis scales to real model sizes on GPU. " * 8000)
    backend.set_device("gpu")

    print(f"{'config':32s} {'params':>8s} {'mem':>7s} {'ms/step':>8s} {'tok/s':>8s}  loss", flush=True)
    for name, dim, layers, heads, kv, mlp, batch, seq in CONFIGS:
        cp.get_default_memory_pool().free_all_blocks()
        ds = LMDataset(ids, seq_len=seq)
        axis.manual_seed(0)
        m = nn.Transformer(vocab_size=32000, dim=dim, n_layers=layers, n_heads=heads,
                           n_kv_heads=kv, mlp_hidden=mlp, max_seq_len=seq).to_gpu()
        opt = optim.AdamW(m.parameters(), lr=3e-4)
        dl = DataLoader(ds, batch_size=batch, shuffle=True, seed=0)
        losses, ms = [], []
        step = 0
        try:
            for inp, tgt in dl:
                if step == 0:  # warmup / compile
                    l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()
                    step = 1
                    continue
                t0 = time.perf_counter()
                l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()
                cp.cuda.Stream.null.synchronize()
                ms.append((time.perf_counter() - t0) * 1000); losses.append(float(l.data))
                step += 1
                if step >= 8:
                    break
            med = float(np.median(ms))
            mem = cp.get_default_memory_pool().used_bytes() / 1e9
            nan = any(np.isnan(x) for x in losses)
            print(f"{name:32s} {m.num_parameters()/1e6:7.0f}M {mem:6.1f}G "
                  f"{med:8.0f} {batch*seq/(med/1000):8.0f}  "
                  f"{losses[0]:.3f}->{losses[-1]:.3f} {'NaN!' if nan else 'ok'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:32s} FAILED: {type(e).__name__}: {str(e)[:80]}", flush=True)
        del m, opt


@app.local_entrypoint()
def main():
    scale.remote()
