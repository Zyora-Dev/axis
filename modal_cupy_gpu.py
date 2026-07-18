"""GPU engine (CuPy / cuBLAS) validation on A100: parity vs CPU + real speedup.
Builds the same model on CPU (numpy) and GPU (model.to_gpu() -> cupy), checks
the logits match, then times training on each and shows the GPU loss curve.

    modal run modal_cupy_gpu.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-cupy-gpu")

DIM, LAYERS, HEADS, KV, MLP = 512, 6, 8, 4, 1024
BATCH, SEQ, STEPS = 8, 128, 25


@app.function(image=image, gpu="A100", timeout=1800)
def run():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import axis
    from axis import nn, optim, backend, ByteTokenizer, LMDataset, DataLoader

    print("cupy available:", backend.has_cupy(), flush=True)

    text = ("Axis trains and fine-tunes transformers on GPU via CuPy and cuBLAS. ") * 500
    ids = ByteTokenizer().encode(text)
    ds = LMDataset(ids, seq_len=SEQ)

    def build():
        axis.manual_seed(0)
        return nn.Transformer(vocab_size=256, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                              n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=256)

    # ---- parity: same input, CPU vs GPU logits ----
    toks = axis.Tensor(np.array([[1, 5, 9, 2, 7, 3, 8, 4]], dtype=np.int64))
    backend.set_device("cpu")
    mcpu = build()
    cpu_logits = mcpu.forward(toks).numpy()
    mgpu = build(); mgpu.to_gpu()
    gpu_logits = mgpu.forward(axis.Tensor(np.array([[1, 5, 9, 2, 7, 3, 8, 4]], dtype=np.int64))).numpy()
    diff = np.abs(cpu_logits - gpu_logits).max()
    print(f"\nPARITY cpu-vs-gpu logits: max|Δ| = {diff:.3e}  match={np.allclose(cpu_logits, gpu_logits, rtol=1e-3, atol=1e-3)}", flush=True)

    def train(on_gpu):
        backend.set_device("gpu" if on_gpu else "cpu")
        m = build()
        if on_gpu:
            m.to_gpu()
        opt = optim.AdamW(m.parameters(), lr=3e-4)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True, seed=0)
        # warmup
        for inp, tgt in dl:
            l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad(); break
        losses, ms = [], []
        step = 0
        for inp, tgt in dl:
            t0 = time.perf_counter()
            l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()
            if on_gpu:
                import cupy as cp; cp.cuda.Stream.null.synchronize()
            dt = time.perf_counter() - t0
            ms.append(dt * 1000); losses.append(float(l.data))
            step += 1
            if step >= STEPS:
                break
        return losses, float(np.median(ms))

    print(f"\nmodel: dim {DIM}, {LAYERS} layers | batch {BATCH} seq {SEQ}", flush=True)
    _, cpu_ms = train(False)
    cpu_toks = BATCH * SEQ / (cpu_ms / 1000)
    print(f"CPU (numpy)     : {cpu_ms:7.0f} ms/step | {cpu_toks:6.0f} tok/s", flush=True)
    losses, gpu_ms = train(True)
    gpu_toks = BATCH * SEQ / (gpu_ms / 1000)
    print(f"GPU (cupy/cuBLAS): {gpu_ms:7.0f} ms/step | {gpu_toks:6.0f} tok/s", flush=True)
    print(f"\n>>> GPU speedup vs CPU: {cpu_ms/gpu_ms:.2f}x  ({cpu_toks:.0f} -> {gpu_toks:.0f} tok/s)", flush=True)

    print("\n=== GPU LOSS CURVE ===", flush=True)
    for i in range(0, len(losses), 4):
        print(f"   step {i:2d}: {losses[i]:.3f}", flush=True)
    print(f"   {losses[0]:.3f} -> {losses[-1]:.3f} ({100*(losses[0]-losses[-1])/losses[0]:.0f}% drop)", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
