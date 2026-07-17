"""Profile WHERE Axis spends time on an A100 — matmul call count/shapes,
GPU-dispatch time vs total step time, and NumPy-fallback share. This tells us
what to fix instead of guessing.

    modal run profile_a100.py
"""
import pathlib

import modal

REPO = pathlib.Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "locomp==1.0.0", "torch")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)

app = modal.App("axis-profile")

MODEL = dict(vocab_size=8192, dim=512, n_layers=6, n_heads=8, n_kv_heads=4,
             mlp_hidden=1024, max_seq_len=256)
BATCH, SEQ = 4, 128


@app.function(image=image, gpu="A100", timeout=1800)
def profile():
    import sys
    import time
    from collections import Counter, defaultdict

    import numpy as np
    sys.path.insert(0, "/root")

    import axis
    from axis import accel, nn, optim
    from axis import ops
    from axis.tensor import Tensor

    print("axis", axis.__version__)
    print("backend:", accel.detect_backend(), "| available:", accel.available())
    accel.enable()

    # ── instrument accel ops: count calls, shapes, and GPU time ────────────
    stats = defaultdict(lambda: [0, 0.0])   # name -> [calls, seconds]
    mm_shapes = Counter()

    def wrap(name, fn):
        def inner(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            stats[name][0] += 1
            stats[name][1] += time.perf_counter() - t
            return r
        return inner

    orig_mm = accel.matmul
    def mm_wrapped(a, b):
        mm_shapes[(a.shape, b.shape)] += 1
        return orig_mm(a, b)
    accel.matmul = wrap("matmul", mm_wrapped)
    accel.softmax_lastdim = wrap("softmax", accel.softmax_lastdim)
    accel.silu = wrap("silu", accel.silu)
    accel.gelu = wrap("gelu", accel.gelu)
    accel.silu_mul = wrap("silu_mul", accel.silu_mul)
    accel.fused_causal_attention = wrap("attention", accel.fused_causal_attention)
    # ops.py imported accel by name — repoint its reference too.
    ops.accel = accel

    axis.manual_seed(0)
    model = nn.Transformer(**MODEL)
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    toks = rng.integers(0, MODEL["vocab_size"], size=(BATCH, SEQ + 1)).astype(np.int64)
    inp, tgt = Tensor(toks[:, :-1]), Tensor(toks[:, 1:])
    print(f"model {model.num_parameters()/1e6:.1f}M | batch {BATCH} seq {SEQ}\n")

    # warmup (compile kernels) — not timed
    loss = model.loss(inp, tgt); loss.backward(); opt.step(); opt.zero_grad()

    stats.clear(); mm_shapes.clear()
    N = 3
    t0 = time.perf_counter()
    for _ in range(N):
        t_fwd = time.perf_counter()
        loss = model.loss(inp, tgt)
        fwd = time.perf_counter() - t_fwd
        t_bwd = time.perf_counter()
        loss.backward()
        bwd = time.perf_counter() - t_bwd
        opt.step(); opt.zero_grad()
    total = (time.perf_counter() - t0) / N

    print(f"per step: {total*1000:.1f} ms  (last fwd {fwd*1000:.1f} / bwd {bwd*1000:.1f})")
    print("\naccel op          calls/step   time/step(ms)   %step")
    for name, (calls, sec) in sorted(stats.items(), key=lambda x: -x[1][1]):
        per = sec / N
        print(f"  {name:14s} {calls/N:8.0f}   {per*1000:10.1f}   {100*per/total:5.1f}%")
    gpu_sec = sum(s[1] for s in stats.values()) / N
    print(f"\n  accel total: {gpu_sec*1000:.1f} ms/step ({100*gpu_sec/total:.0f}% of step)")
    print(f"  rest (numpy autograd/optim): {(total-gpu_sec)*1000:.1f} ms/step ({100*(total-gpu_sec)/total:.0f}%)")

    print("\nmatmul shapes (per step):")
    for (sa, sb), c in mm_shapes.most_common(12):
        print(f"  {c//N:3d}x   {sa} @ {sb}")


@app.local_entrypoint()
def main():
    profile.remote()
