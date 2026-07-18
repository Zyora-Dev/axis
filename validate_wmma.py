"""A100 validation for the tensor-core matmul: does fp16/tensor-core training
converge the same as fp32 scalar, and how much faster? Reliability first —
we don't ship a speedup that changes the loss curve meaningfully.

    modal run validate_wmma.py
"""
import pathlib

import modal

REPO = pathlib.Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "locomp==1.0.0")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)

app = modal.App("axis-validate-wmma")

MODEL = dict(vocab_size=8192, dim=512, n_layers=6, n_heads=8, n_kv_heads=4,
             mlp_hidden=1024, max_seq_len=256)
BATCH, SEQ, STEPS = 4, 128, 8


@app.function(image=image, gpu="A100", timeout=1800)
def validate():
    import sys
    import time

    import numpy as np
    sys.path.insert(0, "/root")

    import axis
    from axis import accel, nn, optim
    from axis.tensor import Tensor

    print("axis", axis.__version__, "| backend", accel.detect_backend(),
          "| available", accel.available(), flush=True)
    accel.enable()

    rng = np.random.default_rng(0)
    toks = rng.integers(0, MODEL["vocab_size"], size=(BATCH, SEQ + 1)).astype(np.int64)
    inp, tgt = Tensor(toks[:, :-1]), Tensor(toks[:, 1:])

    def train(tensor_cores: bool):
        accel.use_tensor_cores(tensor_cores)
        axis.manual_seed(0)
        model = nn.Transformer(**MODEL)
        opt = optim.AdamW(model.parameters(), lr=3e-4)
        # warmup (compile) — not timed
        loss = model.loss(inp, tgt); loss.backward(); opt.step(); opt.zero_grad()
        axis.manual_seed(0)
        model = nn.Transformer(**MODEL)
        opt = optim.AdamW(model.parameters(), lr=3e-4)
        losses = []
        t0 = time.perf_counter()
        for _ in range(STEPS):
            loss = model.loss(inp, tgt)
            loss.backward()
            opt.step(); opt.zero_grad()
            losses.append(float(loss.data))
        dt = (time.perf_counter() - t0) / STEPS
        toks_s = BATCH * SEQ / dt
        return losses, dt * 1000, toks_s

    print("\n-- fp32 scalar tiled (reference numerics) --", flush=True)
    l32, ms32, ts32 = train(False)
    print(f"  step {ms32:.1f} ms | {ts32:.0f} tok/s | loss {l32[0]:.4f} -> {l32[-1]:.4f}", flush=True)

    print("\n-- fp16 tensor cores (wmma) --", flush=True)
    l16, ms16, ts16 = train(True)
    print(f"  step {ms16:.1f} ms | {ts16:.0f} tok/s | loss {l16[0]:.4f} -> {l16[-1]:.4f}", flush=True)

    print("\n== comparison ==", flush=True)
    print(f"  speedup: {ms32/ms16:.2f}x  ({ms32:.0f} -> {ms16:.0f} ms/step, {ts32:.0f} -> {ts16:.0f} tok/s)", flush=True)
    dloss = abs(l16[-1] - l32[-1])
    print(f"  final loss: fp32 {l32[-1]:.4f} vs fp16 {l16[-1]:.4f}  (|Δ|={dloss:.4f})", flush=True)
    print(f"  per-step loss deltas: {[round(a-b,4) for a,b in zip(l16, l32)]}", flush=True)


@app.local_entrypoint()
def main():
    validate.remote()
