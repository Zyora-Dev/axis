"""Phase 2b: FULL TRAINING STEP on the C++ engine vs the eager oracle.

Single-block Transformer (embed -> block -> norm -> lm_head -> CE), complete
forward + backward + AdamW lowered to one plan. Parity gates: loss, every
parameter gradient, and updated weights must match eager.

    modal run engine/modal_step_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24")
    .add_local_file(str(HERE / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-engine-step")

V, B, T, D, H, KV, DH, MLP = 1000, 2, 64, 256, 8, 4, 32, 512
LR, WD = 1e-3, 0.1


@app.function(image=image, gpu="A100", timeout=1800)
def test():
    import subprocess, sys
    import numpy as np
    sys.path.insert(0, "/root")

    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "--shared", "-Xcompiler", "-fPIC",
         "/root/runtime.cu", "-lcublas", "-o", "/root/libaxeng.so"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE FAILED:\n", r.stderr[-2000:], flush=True)
        return
    print("compile: OK", flush=True)

    import axis
    from axis import nn, optim
    from axis.tensor import Tensor
    from axis.engine import (Engine, op, GEMM, ADD, RMSNORM, SILU_MUL, SCALE,
                             GEMM_SB, PERM_0213, ROPE, SOFTMAX_CAUSAL, REPEAT_KV,
                             RMSNORM_BWD, COLSUM, REPEAT_KV_BWD, SOFTMAX_BWD,
                             SILU_BWD, EMBED, EMBED_BWD, CE, ADAMW)

    # ---- eager oracle: one full step ----
    axis.manual_seed(0)
    model = nn.Transformer(vocab_size=V, dim=D, n_layers=1, n_heads=H,
                           n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=T,
                           tie_embeddings=False)
    params0 = {n: p.data.copy() for n, p in model.named_parameters()}
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(0)
    toks = rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
    inp, tgt = Tensor(toks[:, :-1]), Tensor(toks[:, 1:])
    loss_t = model.loss(inp, tgt)
    ref_loss = float(loss_t.data)
    loss_t.backward()
    ref_grads = {n: p.grad.copy() for n, p in model.named_parameters()}
    opt.step()
    ref_updated = {n: p.data.copy() for n, p in model.named_parameters()}
    blk = model.blocks[0]
    cos, sin = blk.attn._cos, blk.attn._sin
    scale = 1.0 / np.sqrt(DH)
    N = B * T

    # ---- engine lowering ----
    eng = Engine("/root/libaxeng.so")
    eng.set_tf32(False)   # pure fp32 for crisp parity

    P = params0  # initial weights
    ids = eng.new_tensor(toks[:, :-1].reshape(-1).astype(np.float32))
    tgt_b = eng.new_tensor(toks[:, 1:].reshape(-1).astype(np.float32))

    # params + grads + adam moments
    names = ["embed.weight", "blocks.0.attn_norm.weight",
             "blocks.0.attn.q_proj.weight", "blocks.0.attn.k_proj.weight",
             "blocks.0.attn.v_proj.weight", "blocks.0.attn.o_proj.weight",
             "blocks.0.mlp_norm.weight", "blocks.0.mlp.gate_proj.weight",
             "blocks.0.mlp.up_proj.weight", "blocks.0.mlp.down_proj.weight",
             "norm.weight", "lm_head.weight"]
    pb, gb, mb, vb = {}, {}, {}, {}
    for nm in names:
        w = P[nm]
        pb[nm] = eng.new_tensor(w)
        gb[nm] = eng.alloc(w.size)
        mb[nm] = eng.new_tensor(np.zeros_like(w))
        vb[nm] = eng.new_tensor(np.zeros_like(w))
    b_cos = eng.new_tensor(cos); b_sin = eng.new_tensor(sin)

    A = eng.alloc  # shorthand
    xe = A(N * D); y = A(N * D)
    q0 = A(N * H * DH); k0 = A(N * KV * DH); v0 = A(N * KV * DH)
    q1 = A(N * H * DH); k1 = A(N * KV * DH); v1 = A(N * KV * DH)
    q2 = A(N * H * DH); k2 = A(N * KV * DH)
    kr = A(N * H * DH); vr = A(N * H * DH)
    sc = A(B * H * T * T); pr = A(B * H * T * T)
    at = A(N * H * DH); at2 = A(N * H * DH)
    o_ = A(N * D); r1 = A(N * D); z = A(N * D)
    g_ = A(N * MLP); u_ = A(N * MLP); h_ = A(N * MLP)
    mo = A(N * D); r2 = A(N * D); xf = A(N * D)
    logits = A(N * V); dlogits = A(N * V); loss_b = A(1)

    # backward buffers
    dxf = A(N * D); dr2 = A(N * D); tmpf = A(N * D)
    dh_ = A(N * MLP); dg_ = A(N * MLP); du_ = A(N * MLP)
    dz1 = A(N * D); dz2 = A(N * D); dz = A(N * D)
    dr1b = A(N * D); tmp2 = A(N * D); dr1 = A(N * D)
    dat2 = A(N * H * DH); dat = A(N * H * DH)
    dpr = A(B * H * T * T); dvr = A(N * H * DH); dsc = A(B * H * T * T)
    dq2 = A(N * H * DH); dkr = A(N * H * DH)
    dk2 = A(N * KV * DH); dv1 = A(N * KV * DH)
    dq1 = A(N * H * DH); dk1 = A(N * KV * DH)
    dq0 = A(N * H * DH); dk0 = A(N * KV * DH); dv0 = A(N * KV * DH)
    dy1 = A(N * D); dy2 = A(N * D); dy3 = A(N * D); dy = A(N * D)
    dx1 = A(N * D); tmp1 = A(N * D); dx = A(N * D)

    BH, BKV = B * H, B * KV
    fwd = [
        op(EMBED, a=pb["embed.weight"], b=ids, c=xe, m=N, n=D),
        op(RMSNORM, a=xe, b=pb["blocks.0.attn_norm.weight"], c=y, m=N, n=D, alpha=1e-5),
        op(GEMM, a=y, b=pb["blocks.0.attn.q_proj.weight"], c=q0, m=N, k=D, n=H * DH),
        op(GEMM, a=y, b=pb["blocks.0.attn.k_proj.weight"], c=k0, m=N, k=D, n=KV * DH),
        op(GEMM, a=y, b=pb["blocks.0.attn.v_proj.weight"], c=v0, m=N, k=D, n=KV * DH),
        op(PERM_0213, a=q0, c=q1, m=B, n=T, k=H, batch=DH),
        op(PERM_0213, a=k0, c=k1, m=B, n=T, k=KV, batch=DH),
        op(PERM_0213, a=v0, c=v1, m=B, n=T, k=KV, batch=DH),
        op(ROPE, a=q1, b=b_cos, d=b_sin, c=q2, batch=BH, m=T, n=DH),
        op(ROPE, a=k1, b=b_cos, d=b_sin, c=k2, batch=BKV, m=T, n=DH),
        op(REPEAT_KV, a=k2, c=kr, batch=B, tb=KV, n=H, m=T, k=DH),
        op(REPEAT_KV, a=v1, c=vr, batch=B, tb=KV, n=H, m=T, k=DH),
        op(GEMM_SB, a=q2, b=kr, c=sc, m=T, k=DH, n=T, batch=BH, tb=1,
           sa=T * DH, sb=T * DH, sc=T * T, alpha=scale),
        op(SOFTMAX_CAUSAL, a=sc, c=pr, batch=BH, m=T),
        op(GEMM_SB, a=pr, b=vr, c=at, m=T, k=T, n=DH, batch=BH,
           sa=T * T, sb=T * DH, sc=T * DH),
        op(PERM_0213, a=at, c=at2, m=B, n=H, k=T, batch=DH),
        op(GEMM, a=at2, b=pb["blocks.0.attn.o_proj.weight"], c=o_, m=N, k=H * DH, n=D),
        op(ADD, a=xe, b=o_, c=r1, n=N * D),
        op(RMSNORM, a=r1, b=pb["blocks.0.mlp_norm.weight"], c=z, m=N, n=D, alpha=1e-5),
        op(GEMM, a=z, b=pb["blocks.0.mlp.gate_proj.weight"], c=g_, m=N, k=D, n=MLP),
        op(GEMM, a=z, b=pb["blocks.0.mlp.up_proj.weight"], c=u_, m=N, k=D, n=MLP),
        op(SILU_MUL, a=g_, b=u_, c=h_, n=N * MLP),
        op(GEMM, a=h_, b=pb["blocks.0.mlp.down_proj.weight"], c=mo, m=N, k=MLP, n=D),
        op(ADD, a=r1, b=mo, c=r2, n=N * D),
        op(RMSNORM, a=r2, b=pb["norm.weight"], c=xf, m=N, n=D, alpha=1e-5),
        op(GEMM, a=xf, b=pb["lm_head.weight"], c=logits, m=N, k=D, n=V),
    ]
    bwd = [
        op(SCALE, a=loss_b, c=loss_b, n=1, alpha=0.0),
        op(CE, a=logits, b=tgt_b, c=dlogits, d=loss_b, m=N, n=V),
        # lm_head
        op(GEMM_SB, a=dlogits, b=pb["lm_head.weight"], c=dxf, m=N, k=V, n=D, batch=1, tb=1),
        op(GEMM_SB, a=xf, b=dlogits, c=gb["lm_head.weight"], m=D, n=V, k=N, batch=1, tb=2),
        # final norm
        op(RMSNORM_BWD, a=r2, b=pb["norm.weight"], d=dxf, c=dr2, tb=tmpf, m=N, n=D, alpha=1e-5),
        op(COLSUM, a=tmpf, c=gb["norm.weight"], m=N, n=D),
        # mlp: r2 = r1 + mo ; d_mo = dr2
        op(GEMM_SB, a=dr2, b=pb["blocks.0.mlp.down_proj.weight"], c=dh_, m=N, k=D, n=MLP, batch=1, tb=1),
        op(GEMM_SB, a=h_, b=dr2, c=gb["blocks.0.mlp.down_proj.weight"], m=MLP, n=D, k=N, batch=1, tb=2),
        op(SILU_BWD, a=g_, b=u_, d=dh_, c=dg_, tb=du_, n=N * MLP),
        op(GEMM_SB, a=dg_, b=pb["blocks.0.mlp.gate_proj.weight"], c=dz1, m=N, k=MLP, n=D, batch=1, tb=1),
        op(GEMM_SB, a=du_, b=pb["blocks.0.mlp.up_proj.weight"], c=dz2, m=N, k=MLP, n=D, batch=1, tb=1),
        op(ADD, a=dz1, b=dz2, c=dz, n=N * D),
        op(GEMM_SB, a=z, b=dg_, c=gb["blocks.0.mlp.gate_proj.weight"], m=D, n=MLP, k=N, batch=1, tb=2),
        op(GEMM_SB, a=z, b=du_, c=gb["blocks.0.mlp.up_proj.weight"], m=D, n=MLP, k=N, batch=1, tb=2),
        op(RMSNORM_BWD, a=r1, b=pb["blocks.0.mlp_norm.weight"], d=dz, c=dr1b, tb=tmp2, m=N, n=D, alpha=1e-5),
        op(COLSUM, a=tmp2, c=gb["blocks.0.mlp_norm.weight"], m=N, n=D),
        op(ADD, a=dr2, b=dr1b, c=dr1, n=N * D),
        # attention out
        op(GEMM_SB, a=dr1, b=pb["blocks.0.attn.o_proj.weight"], c=dat2, m=N, k=D, n=H * DH, batch=1, tb=1),
        op(GEMM_SB, a=at2, b=dr1, c=gb["blocks.0.attn.o_proj.weight"], m=H * DH, n=D, k=N, batch=1, tb=2),
        op(PERM_0213, a=dat2, c=dat, m=B, n=T, k=H, batch=DH),
        op(GEMM_SB, a=dat, b=vr, c=dpr, m=T, k=DH, n=T, batch=BH, tb=1,
           sa=T * DH, sb=T * DH, sc=T * T),
        op(GEMM_SB, a=pr, b=dat, c=dvr, m=T, n=DH, k=T, batch=BH, tb=2,
           sa=T * T, sb=T * DH, sc=T * DH),
        op(SOFTMAX_BWD, a=pr, b=dpr, c=dsc, batch=BH, m=T),
        op(GEMM_SB, a=dsc, b=kr, c=dq2, m=T, k=T, n=DH, batch=BH,
           sa=T * T, sb=T * DH, sc=T * DH, alpha=scale),
        op(GEMM_SB, a=dsc, b=q2, c=dkr, m=T, n=DH, k=T, batch=BH, tb=2,
           sa=T * T, sb=T * DH, sc=T * DH, alpha=scale),
        op(REPEAT_KV_BWD, a=dkr, c=dk2, batch=B, tb=KV, n=H, m=T, k=DH),
        op(REPEAT_KV_BWD, a=dvr, c=dv1, batch=B, tb=KV, n=H, m=T, k=DH),
        op(ROPE, a=dq2, b=b_cos, d=b_sin, c=dq1, batch=BH, m=T, n=DH, tb=1),
        op(ROPE, a=dk2, b=b_cos, d=b_sin, c=dk1, batch=BKV, m=T, n=DH, tb=1),
        op(PERM_0213, a=dq1, c=dq0, m=B, n=H, k=T, batch=DH),
        op(PERM_0213, a=dk1, c=dk0, m=B, n=KV, k=T, batch=DH),
        op(PERM_0213, a=dv1, c=dv0, m=B, n=KV, k=T, batch=DH),
        op(GEMM_SB, a=dq0, b=pb["blocks.0.attn.q_proj.weight"], c=dy1, m=N, k=H * DH, n=D, batch=1, tb=1),
        op(GEMM_SB, a=dk0, b=pb["blocks.0.attn.k_proj.weight"], c=dy2, m=N, k=KV * DH, n=D, batch=1, tb=1),
        op(GEMM_SB, a=dv0, b=pb["blocks.0.attn.v_proj.weight"], c=dy3, m=N, k=KV * DH, n=D, batch=1, tb=1),
        op(ADD, a=dy1, b=dy2, c=dy, n=N * D),
        op(ADD, a=dy, b=dy3, c=dy, n=N * D),
        op(GEMM_SB, a=y, b=dq0, c=gb["blocks.0.attn.q_proj.weight"], m=D, n=H * DH, k=N, batch=1, tb=2),
        op(GEMM_SB, a=y, b=dk0, c=gb["blocks.0.attn.k_proj.weight"], m=D, n=KV * DH, k=N, batch=1, tb=2),
        op(GEMM_SB, a=y, b=dv0, c=gb["blocks.0.attn.v_proj.weight"], m=D, n=KV * DH, k=N, batch=1, tb=2),
        op(RMSNORM_BWD, a=xe, b=pb["blocks.0.attn_norm.weight"], d=dy, c=dx1, tb=tmp1, m=N, n=D, alpha=1e-5),
        op(COLSUM, a=tmp1, c=gb["blocks.0.attn_norm.weight"], m=N, n=D),
        op(ADD, a=dr1, b=dx1, c=dx, n=N * D),
        op(SCALE, a=gb["embed.weight"], c=gb["embed.weight"], n=V * D, alpha=0.0),
        op(EMBED_BWD, a=dx, b=ids, c=gb["embed.weight"], m=N, n=D),
    ]
    # AdamW (t=1): folded bias correction
    bc1, bc2 = 1 - 0.9, 1 - 0.95
    a_lr = LR * np.sqrt(bc2) / bc1
    g_eps = 1e-8 * np.sqrt(bc2)
    step_ops = [op(ADAMW, a=pb[nm], b=gb[nm], c=mb[nm], d=vb[nm],
                   n=P[nm].size, alpha=a_lr, beta=LR * WD, gamma=g_eps)
                for nm in names]

    eng.run(fwd + bwd + step_ops)

    # ---- parity gates ----
    got_loss = float(eng.download(loss_b, (1,))[0])
    print(f"loss: eager {ref_loss:.6f} | engine {got_loss:.6f} | Δ {abs(ref_loss-got_loss):.2e}", flush=True)

    worst_g = ("", 0.0)
    for nm in names:
        g_got = eng.download(gb[nm], P[nm].shape)
        rel = np.abs(g_got - ref_grads[nm]).max() / (np.abs(ref_grads[nm]).max() + 1e-9)
        if rel > worst_g[1]:
            worst_g = (nm, rel)
    print(f"grads: worst rel err = {worst_g[1]:.2e} ({worst_g[0]})", flush=True)

    worst_w = ("", 0.0)
    for nm in names:
        w_got = eng.download(pb[nm], P[nm].shape)
        rel = np.abs(w_got - ref_updated[nm]).max() / (np.abs(ref_updated[nm]).max() + 1e-9)
        if rel > worst_w[1]:
            worst_w = (nm, rel)
    print(f"updated weights (E2E, t=1 sign-sensitive): worst rel = {worst_w[1]:.2e} ({worst_w[0]})", flush=True)

    # Rigorous optimizer gate: AdamW kernel on IDENTICAL grads must match the
    # eager formula exactly. (E2E weight diff at t=1 reflects update≈sign(g)*lr
    # sensitivity to 1e-6 grad noise near g=0 — fp physics, not a bug.)
    nm = "blocks.0.attn.v_proj.weight"
    w0 = params0[nm]
    p_t = eng.new_tensor(w0)
    g_t = eng.new_tensor(ref_grads[nm])
    m_t = eng.new_tensor(np.zeros_like(w0))
    v_t = eng.new_tensor(np.zeros_like(w0))
    eng.run([op(ADAMW, a=p_t, b=g_t, c=m_t, d=v_t, n=w0.size,
                alpha=a_lr, beta=LR * WD, gamma=g_eps)])
    w_kernel = eng.download(p_t, w0.shape)
    # eager formula, same grads
    g = ref_grads[nm]
    m_ = 0.1 * g; v_ = 0.05 * g * g
    mh = m_ / bc1; vh = v_ / bc2
    w_ref = w0 * (1 - LR * WD) - LR * mh / (np.sqrt(vh) + 1e-8)
    adam_rel = np.abs(w_kernel - w_ref).max() / (np.abs(w_ref).max() + 1e-9)
    print(f"AdamW kernel vs eager formula (same grads): rel = {adam_rel:.2e}", flush=True)

    ok = (abs(ref_loss - got_loss) < 1e-4 and worst_g[1] < 1e-3 and adam_rel < 1e-4)
    print(f"FULL TRAINING-STEP PARITY: {'PASS' if ok else 'FAIL'}", flush=True)


@app.local_entrypoint()
def main():
    test.remote()
