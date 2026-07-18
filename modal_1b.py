"""Measure Axis at 1B scale — the aq-5b Phase-3 class shape (1.26B params:
hidden 1536, 48 layers, 24 heads / 8 kv, mlp 4096, seq 2048), fp16 AMP +
gradient checkpointing, A100-80GB.

    modal run modal_1b.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-1b")
VOCAB, DIM, LAYERS, HEADS, KV, MLP, SEQ = 32000, 1536, 48, 24, 8, 4096, 2048


@app.function(image=image, gpu="A100-80GB", timeout=2400)
def measure():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import cupy as cp
    import axis
    from axis import nn, optim, backend

    backend.set_device("gpu")

    def run(batch, fp16, ckpt, steps=4):
        cp.get_default_memory_pool().free_all_blocks()
        try:
            axis.manual_seed(0)
            m = nn.Transformer(vocab_size=VOCAB, dim=DIM, n_layers=LAYERS,
                               n_heads=HEADS, n_kv_heads=KV, mlp_hidden=MLP,
                               max_seq_len=SEQ, tie_embeddings=True)
            n_params = m.num_parameters()
            m.to_gpu(fp16=fp16)
            m.grad_checkpoint = ckpt
            opt = optim.AdamW(m.parameters(), lr=2e-4)
            scaler = optim.GradScaler() if fp16 else None
            rng = np.random.default_rng(0)
            toks = rng.integers(0, VOCAB, size=(batch, SEQ + 1)).astype(np.int64)
            inp, tgt = axis.Tensor(toks[:, :-1]), axis.Tensor(toks[:, 1:])

            def step():
                loss = m.loss(inp, tgt)
                lv = float(loss.data)
                if fp16:
                    scaler.scale(loss).backward(); scaler.step(opt)
                else:
                    loss.backward(); opt.step()
                opt.zero_grad()
                return lv

            step()  # warmup
            cp.cuda.Stream.null.synchronize()
            ms, losses = [], []
            for _ in range(steps):
                t0 = time.perf_counter()
                losses.append(step())
                cp.cuda.Stream.null.synchronize()
                ms.append((time.perf_counter() - t0) * 1000)
            med = float(np.median(ms))
            mem = cp.get_default_memory_pool().total_bytes() / 1e9
            print(f"  {n_params/1e9:.2f}B | batch {batch:2d} fp16={fp16} ckpt={ckpt}: "
                  f"{med:6.0f} ms | {batch*SEQ/(med/1000):6.0f} tok/s | pool {mem:5.1f}G | "
                  f"loss {losses[0]:.3f}->{losses[-1]:.3f}", flush=True)
            del m, opt
        except Exception as e:  # noqa: BLE001
            print(f"  batch {batch:2d} fp16={fp16} ckpt={ckpt}: FAILED "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
        finally:
            cp.get_default_memory_pool().free_all_blocks()

    print(f"1B-class Llama (hidden {DIM}, {LAYERS} layers, seq {SEQ}) — A100-80GB\n", flush=True)
    run(4, True, True)
    run(8, True, True)
    run(16, True, True)


@app.local_entrypoint()
def main():
    measure.remote()
