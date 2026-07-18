"""A100 validation of gradient checkpointing: peak memory + tok/s with/without,
and the payoff — a batch size that only fits WITH checkpointing.

    modal run modal_gradckpt.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-gradckpt")
VOCAB, DIM, LAYERS, HEADS, KV, MLP, SEQ = 32000, 768, 12, 12, 4, 2048, 512


@app.function(image=image, gpu="A100", timeout=2400)
def validate():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import cupy as cp
    import axis
    from axis import nn, optim, backend

    backend.set_device("gpu")

    def run(ckpt, batch, steps=6):
        cp.get_default_memory_pool().free_all_blocks()
        axis.manual_seed(0)
        m = nn.Transformer(vocab_size=VOCAB, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                           n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=SEQ,
                           tie_embeddings=False).to_gpu()
        m.grad_checkpoint = ckpt
        opt = optim.AdamW(m.parameters(), lr=3e-4)
        rng = np.random.default_rng(0)
        toks = rng.integers(0, VOCAB, size=(batch, SEQ + 1)).astype(np.int64)
        inp, tgt = axis.Tensor(toks[:, :-1]), axis.Tensor(toks[:, 1:])
        try:
            l = m.loss(inp, tgt); l.backward(); opt.step(); opt.zero_grad()  # warmup
            cp.get_default_memory_pool().free_all_blocks()
            l = m.loss(inp, tgt)
            cp.cuda.Stream.null.synchronize()
            peak_fwd = cp.get_default_memory_pool().used_bytes() / 1e9
            l.backward(); opt.step(); opt.zero_grad()
            ms, losses = [], []
            for _ in range(steps):
                t0 = time.perf_counter()
                l = m.loss(inp, tgt)
                lv = float(l.data)
                l.backward(); opt.step(); opt.zero_grad()
                cp.cuda.Stream.null.synchronize()
                ms.append((time.perf_counter() - t0) * 1000); losses.append(lv)
            med = float(np.median(ms))
            print(f"  ckpt={str(ckpt):5s} batch={batch:3d}: {med:6.0f} ms | "
                  f"{batch*SEQ/(med/1000):6.0f} tok/s | fwd-peak {peak_fwd:5.1f}G | "
                  f"loss {losses[0]:.3f}->{losses[-1]:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ckpt={str(ckpt):5s} batch={batch:3d}: FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
        finally:
            del m, opt
            cp.get_default_memory_pool().free_all_blocks()

    print(f"125M Llama seq {SEQ}, A100 40GB\n", flush=True)
    run(False, 8)
    run(True, 8)
    run(False, 64)     # big batch without ckpt — expect heavy memory or OOM
    run(True, 64)      # big batch with ckpt — the payoff
    run(True, 96)      # push further, only possible with ckpt


@app.local_entrypoint()
def main():
    validate.remote()
