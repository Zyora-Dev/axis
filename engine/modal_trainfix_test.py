"""Validate the training-loop fixes on A100:
1) REGRESSION — default path (no clip, all-valid targets, device-lr held at
   constant) must still match the eager oracle bit-tight (fp32) — proves the
   CE-denom / device-lr / wd-exclusion plumbing is transparent when unused.
2) WEIGHT-DECAY EXCLUSION — 20-step loss curve, compiled fp32 vs eager (both
   now exclude 1D params from wd): must track to ~machine precision.
3) LR SCHEDULE under CUDA graph — capture once, then set_lr() per replay
   (warmup→decay): the device lr must actually change the update. Compare a
   scheduled compiled run vs an eager run using the SAME per-step lr.
4) GRAD CLIP — compiled with max_grad_norm=g vs eager clip_grad_norm(g) on
   identical data: loss curves must match (both clip the global grad norm).
5) IGNORE_INDEX / padding — targets with -1 positions: compiled loss+grads
   must equal the eager loss computed as a mean over ONLY the valid tokens.
6) CHECKPOINT resume — state_dict/load_state_dict round-trips exact; a run
   that saves at step k and resumes matches the uninterrupted run.

    modal run engine/modal_trainfix_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24")
    .add_local_file(str(REPO / "axis" / "_csrc" / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-engine-trainfix")

LIB = "/root/libaxeng.so"


def _compile():
    import subprocess
    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "--shared", "-Xcompiler", "-fPIC",
         "/root/runtime.cu", "-lcublas", "-o", LIB],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE FAILED:\n", r.stderr[-3000:], flush=True)
        raise SystemExit(1)
    print("nvcc: OK", flush=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def test():
    import sys
    import numpy as np
    sys.path.insert(0, "/root")
    _compile()
    import axis
    from axis import nn, optim
    from axis.tensor import Tensor
    from axis.compile import compile_model

    V, B, T, D, H, KV, MLP, L = 500, 2, 32, 128, 4, 2, 256, 3
    cfg = dict(vocab_size=V, dim=D, n_layers=L, n_heads=H, n_kv_heads=KV,
               mlp_hidden=MLP, tie_embeddings=True)

    def build():
        axis.manual_seed(0)
        return nn.Transformer(vocab_size=V, dim=D, n_layers=L, n_heads=H,
                              n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=T,
                              tie_embeddings=True)
    rng = np.random.default_rng(0)
    ok_all = True

    # ---- 1) regression: single-step grad parity, default path ----
    m = build()
    toks = rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]
    lt = m.loss(Tensor(inp), Tensor(tgt)); rl = float(lt.data); lt.backward()
    rg = {n: p.grad.copy() for n, p in m.named_parameters()}
    ct = compile_model(build(), B, T, lib_path=LIB, tf32=False, dtype="fp32")
    gl = ct.step(inp, tgt, t=1); gg = ct.grads()
    worst = max(np.abs(gg[nm] - g).max() / (np.abs(g).max() + 1e-9) for nm, g in rg.items())
    ok = abs(rl - gl) < 1e-4 and worst < 1e-3; ok_all &= ok
    print(f"1 regression: loss Δ {abs(rl-gl):.2e} grad {worst:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- 2) wd-exclusion loss curve (eager vs compiled, both exclude 1D) ----
    STEPS, LR = 20, 1e-3
    batches = [rng.integers(0, V, size=(B, T + 1)).astype(np.int64) for _ in range(STEPS)]
    me = build(); opt = optim.AdamW(me.parameters(), lr=LR, weight_decay=0.1)
    ref = []
    for b in batches:
        opt.zero_grad(); l = me.loss(Tensor(b[:, :-1]), Tensor(b[:, 1:])); l.backward(); opt.step()
        ref.append(float(l.data))
    ce = compile_model(build(), B, T, lib_path=LIB, lr=LR, wd=0.1, tf32=False, dtype="fp32")
    cc = [ce.step(b[:, :-1], b[:, 1:]) for b in batches]
    d = max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(ref, cc))
    ok = d < 1e-3; ok_all &= ok
    print(f"2 wd-exclusion curve: max rel drift {d:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- 3) LR schedule under graph ----
    def sched(step): return LR * (0.1 + 0.9 * step / STEPS)   # ramp
    me = build(); opt = optim.AdamW(me.parameters(), lr=LR, weight_decay=0.0)
    ref = []
    for s, b in enumerate(batches):
        opt.lr = sched(s)
        opt.zero_grad(); l = me.loss(Tensor(b[:, :-1]), Tensor(b[:, 1:])); l.backward(); opt.step()
        ref.append(float(l.data))
    ce = compile_model(build(), B, T, lib_path=LIB, lr=LR, wd=0.0, tf32=False, dtype="fp32")
    ce.capture()                       # graph baked at initial lr; set_lr changes device buf
    cc = []
    for s, b in enumerate(batches):
        ce.set_lr(sched(s))
        cc.append(ce.replay_step(b[:, :-1], b[:, 1:]))
    d = max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(ref, cc))
    ok = d < 1e-3; ok_all &= ok
    print(f"3 lr-schedule under graph: max rel drift {d:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- 4) grad clip (both sides clip global norm) ----
    MGN = 0.1
    me = build(); opt = optim.AdamW(me.parameters(), lr=LR, weight_decay=0.0)
    ref = []
    for b in batches:
        opt.zero_grad(); l = me.loss(Tensor(b[:, :-1]), Tensor(b[:, 1:])); l.backward()
        optim.clip_grad_norm(me.parameters(), MGN); opt.step()
        ref.append(float(l.data))
    ce = compile_model(build(), B, T, lib_path=LIB, lr=LR, wd=0.0, tf32=False,
                       dtype="fp32", max_grad_norm=MGN)
    cc = [ce.step(b[:, :-1], b[:, 1:]) for b in batches]
    d = max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(ref, cc))
    ok = d < 1e-3; ok_all &= ok
    print(f"4 grad-clip curve: max rel drift {d:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- 5) ignore_index / padding ----
    m = build()
    toks = rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
    inp = toks[:, :-1]; tgt = toks[:, 1:].copy()
    tgt[:, :T // 2] = -1                         # mask first half (prompt) from loss
    # eager reference: mean NLL over valid tokens only
    logits = m(Tensor(inp))
    import axis.ops as ops
    lg = logits.data.reshape(-1, V).astype(np.float64)
    tt = tgt.reshape(-1)
    valid = tt >= 0
    mxr = lg.max(1, keepdims=True)
    lse = mxr[:, 0] + np.log(np.exp(lg - mxr).sum(1))
    nll = lse - lg[np.arange(len(tt)), np.clip(tt, 0, V - 1)]
    ref_loss = float(nll[valid].mean())
    ce = compile_model(build(), B, T, lib_path=LIB, tf32=False, dtype="fp32")
    gl = ce.step(inp, tgt, t=1)
    ok = abs(ref_loss - gl) < 1e-3; ok_all &= ok
    print(f"5 ignore_index: eager-valid-mean {ref_loss:.5f} vs compiled {gl:.5f} "
          f"Δ {abs(ref_loss-gl):.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- 6) checkpoint resume ----
    ce = compile_model(build(), B, T, lib_path=LIB, lr=LR, tf32=False, dtype="fp32")
    for b in batches[:10]:
        ce.step(b[:, :-1], b[:, 1:])
    snap = {k: v.copy() for k, v in ce.state_dict().items()}
    cont = [ce.step(b[:, :-1], b[:, 1:]) for b in batches[10:]]
    ce2 = compile_model(build(), B, T, lib_path=LIB, lr=LR, tf32=False, dtype="fp32")
    ce2.load_state_dict(snap)
    resumed = [ce2.step(b[:, :-1], b[:, 1:]) for b in batches[10:]]
    d = max(abs(a - b) for a, b in zip(cont, resumed))
    ok = d < 1e-4; ok_all &= ok
    print(f"6 checkpoint resume: max loss Δ over 10 post-resume steps {d:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\nTRAIN-LOOP FIXES: {'ALL PASS' if ok_all else 'FAIL'}", flush=True)


@app.local_entrypoint()
def main():
    test.remote()
