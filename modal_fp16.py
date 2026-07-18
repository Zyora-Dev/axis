"""A100 validation of fp16 storage mode (AMP): speed + convergence vs fp32 on
the 125M benchmark model.

    modal run modal_fp16.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-fp16")
VOCAB, DIM, LAYERS, HEADS, KV, MLP = 32000, 768, 12, 12, 4, 2048
BATCH, SEQ, STEPS = 8, 512, 15


@app.function(image=image, gpu="A100", timeout=2400)
def validate():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import cupy as cp
    import axis
    from axis import nn, optim, backend
    from axis.tensor import set_fp16_mode

    rng = np.random.default_rng(0)
    toks = rng.integers(0, VOCAB, size=(BATCH, SEQ + 1)).astype(np.int64)
    inp_np, tgt_np = toks[:, :-1], toks[:, 1:]

    def run(fp16):
        set_fp16_mode(fp16)
        backend.set_device("cpu")   # build in fp32 on host, then move
        axis.manual_seed(0)
        m = nn.Transformer(vocab_size=VOCAB, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                           n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=SEQ,
                           tie_embeddings=False)
        m.to_gpu(fp16=fp16)
        opt = optim.AdamW(m.parameters(), lr=3e-4)
        scaler = optim.GradScaler() if fp16 else None
        inp, tgt = axis.Tensor(inp_np), axis.Tensor(tgt_np)
        losses, ms = [], []
        for i in range(STEPS + 2):
            if i == 2:
                cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
            loss = m.loss(inp, tgt)
            lval = float(loss.data)
            if fp16:
                scaler.scale(loss).backward()
                scaler.step(opt)
            else:
                loss.backward(); opt.step()
            opt.zero_grad()
            if i >= 2:
                losses.append(lval)
        cp.cuda.Stream.null.synchronize()
        med = (time.perf_counter() - t0) / STEPS * 1000
        cp.get_default_memory_pool().free_all_blocks()
        l = m.loss(inp, tgt)
        cp.cuda.Stream.null.synchronize()
        peak = cp.get_default_memory_pool().used_bytes() / 1e9
        del m, opt, l
        cp.get_default_memory_pool().free_all_blocks()
        label = "fp16 (AMP)" if fp16 else "fp32+TF32 "
        print(f"{label}: {med:6.0f} ms/step | {BATCH*SEQ/(med/1000):6.0f} tok/s | "
              f"peak {peak:.1f}G | loss {losses[0]:.3f}->{losses[-1]:.3f}", flush=True)
        return med, losses

    print(f"125M Llama, batch {BATCH} seq {SEQ}, A100\n", flush=True)
    ms32, l32 = run(False)
    ms16, l16 = run(True)
    print(f"\nfp16 speedup: {ms32/ms16:.2f}x", flush=True)
    print(f"loss curves (fp32 vs fp16): "
          f"{[round(a,3) for a in l32[:6]]} vs {[round(b,3) for b in l16[:6]]}", flush=True)


@app.local_entrypoint()
def main():
    validate.remote()
