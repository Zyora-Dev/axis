"""Instrumented A100 run: matmul-only residency. Separates fwd/bwd time and
reports uploads vs cache-hits, so we can see if residency helps or regresses
and why. Compares residency ON vs OFF, same model.

    modal run resident_bench.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "locomp==1.0.0")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-resident-bench")
MODEL = dict(vocab_size=8192, dim=512, n_layers=6, n_heads=8, n_kv_heads=4,
             mlp_hidden=1024, max_seq_len=256)
BATCH, SEQ, STEPS = 4, 128, 6


@app.function(image=image, gpu="A100", timeout=1800)
def run():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import axis
    from axis import accel, nn, optim
    from axis.tensor import Tensor

    print("axis", axis.__version__, "| backend", accel.detect_backend(), flush=True)
    accel.enable()
    rng = np.random.default_rng(0)
    toks = rng.integers(0, MODEL["vocab_size"], size=(BATCH, SEQ + 1)).astype(np.int64)
    inp, tgt = Tensor(toks[:, :-1]), Tensor(toks[:, 1:])

    def bench(resident):
        accel._RESIDENT = resident
        axis.manual_seed(0)
        model = nn.Transformer(**MODEL)
        opt = optim.AdamW(model.parameters(), lr=3e-4)
        # warmup / compile
        loss = model.loss(inp, tgt); loss.backward(); opt.step(); opt.zero_grad()
        fwd = bwd = 0.0
        accel.reset_residency_stats()
        for _ in range(STEPS):
            t = time.perf_counter()
            loss = model.loss(inp, tgt)
            fwd += time.perf_counter() - t
            t = time.perf_counter()
            loss.backward()
            bwd += time.perf_counter() - t
            opt.step(); opt.zero_grad()
        st = accel.residency_stats()
        step = (fwd + bwd) / STEPS
        print(f"  resident={resident}: step {step*1000:.0f} ms "
              f"(fwd {fwd/STEPS*1000:.0f} / bwd {bwd/STEPS*1000:.0f}) "
              f"| {BATCH*SEQ/step:.0f} tok/s | uploads/step {st['uploads']//STEPS} "
              f"hits/step {st['hits']//STEPS} | loss->{float(loss.data):.4f}", flush=True)

    bench(False)
    bench(True)


@app.local_entrypoint()
def main():
    run.remote()
