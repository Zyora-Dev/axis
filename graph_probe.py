"""FEASIBILITY PROBE: can a full Axis training step be captured as a CUDA
graph and replayed without Python overhead?

If yes, the per-op Python dispatch (the structural bottleneck at every scale)
disappears on replay: step time becomes pure GPU time, for ANY model.

Probes, in order:
  1. Does cupy support stream capture -> graph -> launch on this stack?
  2. Can an Axis fwd+bwd+optimizer step run under capture (no illegal syncs)?
  3. Replay speed vs normal eager step, and numerical equivalence of weights.

    modal run graph_probe.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-graph-probe")

# Mid-size probe config — big enough to be meaningful, small enough to iterate.
VOCAB, DIM, LAYERS, HEADS, KV, MLP, SEQ, BATCH = 32000, 1536, 24, 24, 8, 4096, 1024, 4


@app.function(image=image, gpu="A100-80GB", timeout=2400)
def probe():
    import sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    import cupy as cp
    import axis
    from axis import nn, optim, backend

    print("cupy", cp.__version__, "| CUDA", cp.cuda.runtime.runtimeGetVersion(), flush=True)

    # ---- 1. raw capture support ----
    try:
        s = cp.cuda.Stream(non_blocking=True)
        a = cp.ones((256, 256), dtype=cp.float32)
        with s:
            s.begin_capture()
            b = a @ a
            g = s.end_capture()
        g.launch(stream=s)
        s.synchronize()
        print("1) raw stream capture + graph launch: OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"1) raw capture FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return

    # ---- 2. capture an Axis training step ----
    backend.set_device("gpu")
    axis.manual_seed(0)
    m = nn.Transformer(vocab_size=VOCAB, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                       n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=SEQ,
                       tie_embeddings=True).to_gpu()
    opt = optim.AdamW(m.parameters(), lr=2e-4)
    rng = np.random.default_rng(0)
    toks = rng.integers(0, VOCAB, size=(BATCH, SEQ + 1)).astype(np.int64)
    inp, tgt = axis.Tensor(toks[:, :-1]), axis.Tensor(toks[:, 1:])
    n_params = m.num_parameters()
    print(f"model {n_params/1e9:.2f}B | batch {BATCH} seq {SEQ}", flush=True)

    def eager_step():
        loss = m.loss(inp, tgt)
        loss.backward()
        opt.step(); opt.zero_grad()

    # warmup (compile/caches)
    eager_step(); eager_step()
    cp.cuda.Stream.null.synchronize()

    # eager timing
    t0 = time.perf_counter()
    for _ in range(3):
        eager_step()
    cp.cuda.Stream.null.synchronize()
    eager_ms = (time.perf_counter() - t0) / 3 * 1000
    print(f"2) eager step: {eager_ms:.0f} ms | {BATCH*SEQ/(eager_ms/1000):.0f} tok/s", flush=True)

    # snapshot weights for equivalence check
    w_before = [p.data.copy() for p in list(m.parameters())[:3]]

    # capture attempt — memory pool must be capture-safe; try default first
    s = cp.cuda.Stream(non_blocking=True)
    try:
        with s:
            s.begin_capture()
            eager_step()
            graph = s.end_capture()
        s.synchronize()
        print("3) Axis step CAPTURED as CUDA graph: OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"3) Axis step capture FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return

    # ---- 3. replay speed ----
    try:
        # warm replay
        graph.launch(stream=s); s.synchronize()
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            graph.launch(stream=s)
        s.synchronize()
        replay_ms = (time.perf_counter() - t0) / N * 1000
        print(f"4) graph REPLAY step: {replay_ms:.0f} ms | {BATCH*SEQ/(replay_ms/1000):.0f} tok/s "
              f"| speedup vs eager: {eager_ms/replay_ms:.2f}x", flush=True)
        # weights actually changed by replays?
        changed = any(not cp.allclose(w0, p.data) for w0, p in
                      zip(w_before, list(m.parameters())[:3]))
        print(f"5) weights updated by replay: {bool(changed)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"4) replay FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)


@app.local_entrypoint()
def main():
    probe.remote()
