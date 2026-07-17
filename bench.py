"""Axis cross-vendor benchmark — NVIDIA (CUDA) + AMD (ROCm) via Modal.

Measures how the Axis framework itself performs: forward latency, training
step time, and tokens/sec on a realistic transformer, on real datacenter GPUs.
A PyTorch run of the same model is included as a reference point (where we
stand), not a competition.

    modal run bench.py                 # runs NVIDIA + AMD
    modal run bench.py --vendor nvidia # just one
"""
import pathlib

import modal

REPO = pathlib.Path(__file__).parent


def _image(extra_pip=()):
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
        .pip_install("numpy>=1.24", "locomp==1.0.0", *extra_pip)
        .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
    )


# NVIDIA image includes a CUDA PyTorch (reference). AMD image: ROCm devel + torch-rocm.
nvidia_image = _image(extra_pip=("torch",))
amd_image = (
    modal.Image.from_registry("rocm/dev-ubuntu-22.04:6.2-complete", add_python="3.11")
    .pip_install("numpy>=1.24", "locomp==1.0.0")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)

app = modal.App("axis-bench")

MODEL = dict(vocab_size=8192, dim=512, n_layers=6, n_heads=8, n_kv_heads=4,
             mlp_hidden=1024, max_seq_len=256)
BATCH, SEQ, STEPS = 4, 128, 12


def _run_axis_bench(label: str):
    import sys
    import time

    import numpy as np
    sys.path.insert(0, "/root")

    import axis
    from axis import accel, nn, optim
    from axis.tensor import Tensor

    print(f"\n{'='*66}\nAXIS BENCHMARK — {label}\n{'='*66}")
    print("axis", axis.__version__)
    be = accel.detect_backend()
    ok = accel.available()
    print(f"detected backend: {be} | available: {ok}")
    if not ok:
        print("!! GPU not available — reference engine only. Skipping GPU bench.")
        return {"label": label, "backend": be, "available": False}
    accel.enable()

    axis.manual_seed(0)
    model = nn.Transformer(**MODEL)
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    toks = rng.integers(0, MODEL["vocab_size"], size=(BATCH, SEQ + 1)).astype(np.int64)
    inp, tgt = Tensor(toks[:, :-1]), Tensor(toks[:, 1:])
    nparam = model.num_parameters()
    print(f"model: {nparam/1e6:.1f}M params | batch {BATCH} seq {SEQ}")

    # forward-only latency
    def fwd():
        with axis.no_grad():
            model.loss(inp, tgt)
    fwd(); fwd()  # warmup (compile kernels)
    t = time.perf_counter()
    for _ in range(5):
        fwd()
    fwd_ms = (time.perf_counter() - t) / 5 * 1000

    # training step
    def step():
        model.zero_grad()
        loss = model.loss(inp, tgt)
        loss.backward()
        optim.clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        return loss.item()
    step(); step()  # warmup
    t = time.perf_counter()
    losses = [step() for _ in range(STEPS)]
    step_ms = (time.perf_counter() - t) / STEPS * 1000
    toks_per_s = (BATCH * SEQ) / (step_ms / 1000)

    print(f"forward latency : {fwd_ms:9.2f} ms")
    print(f"train step      : {step_ms:9.2f} ms")
    print(f"throughput      : {toks_per_s:9.1f} tokens/s")
    print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    return {"label": label, "backend": be, "available": True, "params_M": nparam/1e6,
            "fwd_ms": fwd_ms, "step_ms": step_ms, "toks_per_s": toks_per_s}


def _run_torch_reference(label: str):
    import time

    import numpy as np
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        print("(torch not available — skipping reference)")
        return None
    if not torch.cuda.is_available():
        print("(torch.cuda unavailable — skipping reference)")
        return None
    dev = "cuda"
    print(f"\n--- PyTorch reference ({label}) — same model shape ---")

    # Minimal equivalent decoder for a fair-shape reference.
    class Block(torch.nn.Module):
        def __init__(s, d, h):
            super().__init__()
            s.n = torch.nn.LayerNorm(d); s.q = torch.nn.Linear(d, d, bias=False)
            s.k = torch.nn.Linear(d, d, bias=False); s.v = torch.nn.Linear(d, d, bias=False)
            s.o = torch.nn.Linear(d, d, bias=False); s.h = h
            s.n2 = torch.nn.LayerNorm(d)
            s.f1 = torch.nn.Linear(d, MODEL["mlp_hidden"]); s.f2 = torch.nn.Linear(MODEL["mlp_hidden"], d)
        def forward(s, x):
            B, T, D = x.shape; H = s.h; hd = D // H
            y = s.n(x)
            q = s.q(y).view(B, T, H, hd).transpose(1, 2)
            k = s.k(y).view(B, T, H, hd).transpose(1, 2)
            v = s.v(y).view(B, T, H, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
            x = x + s.f2(F.silu(s.f1(s.n2(x))))
            return x

    class M(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = torch.nn.Embedding(MODEL["vocab_size"], MODEL["dim"])
            s.blocks = torch.nn.ModuleList([Block(MODEL["dim"], MODEL["n_heads"]) for _ in range(MODEL["n_layers"])])
            s.nf = torch.nn.LayerNorm(MODEL["dim"])
        def forward(s, t):
            x = s.emb(t)
            for b in s.blocks: x = b(x)
            return x @ s.emb.weight.T

    m = M().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    toks = torch.tensor(rng.integers(0, MODEL["vocab_size"], (BATCH, SEQ + 1)), device=dev)
    inp, tgt = toks[:, :-1], toks[:, 1:]

    def step():
        opt.zero_grad()
        logits = m(inp)
        loss = F.cross_entropy(logits.reshape(-1, MODEL["vocab_size"]), tgt.reshape(-1))
        loss.backward(); opt.step()
        torch.cuda.synchronize()
        return loss.item()
    step(); step()
    t = time.perf_counter()
    for _ in range(STEPS): step()
    step_ms = (time.perf_counter() - t) / STEPS * 1000
    print(f"torch train step: {step_ms:9.2f} ms | {(BATCH*SEQ)/(step_ms/1000):9.1f} tokens/s")
    return {"label": label + " (torch ref)", "step_ms": step_ms}


@app.function(image=nvidia_image, gpu="A100", timeout=1800)
def bench_nvidia():
    r = _run_axis_bench("NVIDIA A100 (CUDA)")
    ref = _run_torch_reference("NVIDIA A100")
    return r, ref


# AMD registered only when requested (avoids MI300X validation blocking NVIDIA
# runs on accounts without AMD access).
import os as _os

if _os.environ.get("AXIS_AMD") == "1":
    @app.function(image=amd_image, gpu=_os.environ.get("AXIS_AMD_GPU", "MI300X"), timeout=1800)
    def bench_amd():
        return _run_axis_bench("AMD MI300X (ROCm)"), None


@app.local_entrypoint()
def main(vendor: str = "both"):
    results = []
    if vendor in ("both", "nvidia"):
        results.append(bench_nvidia.remote())
    if vendor in ("both", "amd") and _os.environ.get("AXIS_AMD") == "1":
        try:
            results.append(bench_amd.remote())
        except Exception as e:
            print(f"AMD run unavailable: {e}")

    print("\n" + "=" * 66)
    print("SUMMARY — Axis training performance (cross-vendor)")
    print("=" * 66)
    for axis_r, ref in results:
        if axis_r and axis_r.get("available"):
            print(f"{axis_r['label']:28s}  step {axis_r['step_ms']:8.2f} ms  "
                  f"{axis_r['toks_per_s']:8.1f} tok/s  ({axis_r['params_M']:.1f}M)")
        elif axis_r:
            print(f"{axis_r['label']:28s}  backend={axis_r['backend']}  (GPU unavailable)")
        if ref:
            print(f"{ref['label']:28s}  step {ref['step_ms']:8.2f} ms  (reference)")
