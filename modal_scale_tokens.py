"""tok/s scaling on the 125M benchmark model — batch sweep now that peak
memory is below PyTorch. Shows how the freed memory converts to throughput.

    modal run modal_scale_tokens.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-scale-tokens")
VOCAB, DIM, LAYERS, HEADS, KV, MLP, SEQ = 32000, 768, 12, 12, 4, 2048, 512


@app.function(image=image, gpu="A100", timeout=2400)
def sweep():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import cupy as cp
    import axis
    from axis import nn, optim, backend

    backend.set_device("gpu")
    print(f"125M Llama, seq {SEQ} — tok/s vs batch size (A100 40GB)", flush=True)
    print(f"{'batch':>6s} {'ms/step':>8s} {'tok/s':>8s} {'peak mem':>9s}", flush=True)
    for batch in (8, 16, 32, 48, 64):
        cp.get_default_memory_pool().free_all_blocks()
        try:
            axis.manual_seed(0)
            m = nn.Transformer(vocab_size=VOCAB, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                               n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=SEQ,
                               tie_embeddings=False).to_gpu()
            opt = optim.AdamW(m.parameters(), lr=3e-4)
            rng = np.random.default_rng(0)
            toks = rng.integers(0, VOCAB, size=(batch, SEQ + 1)).astype(np.int64)
            inp, tgt = axis.Tensor(toks[:, :-1]), axis.Tensor(toks[:, 1:])
            # warmup
            l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()
            # peak = end of forward
            cp.get_default_memory_pool().free_all_blocks()
            l = m.loss(inp, tgt)
            cp.cuda.Stream.null.synchronize()
            peak = cp.get_default_memory_pool().used_bytes() / 1e9
            l.backward(); opt.step(); opt.zero_grad()
            ms = []
            for _ in range(6):
                t0 = time.perf_counter()
                l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()
                cp.cuda.Stream.null.synchronize()
                ms.append((time.perf_counter() - t0) * 1000)
            med = float(np.median(ms))
            print(f"{batch:6d} {med:8.0f} {batch*SEQ/(med/1000):8.0f} {peak:8.1f}G", flush=True)
            del m, opt
        except Exception as e:  # noqa: BLE001
            print(f"{batch:6d}  FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
            break


@app.local_entrypoint()
def main():
    sweep.remote()
