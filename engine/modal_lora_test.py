"""Compiled LoRA fine-tuning path validation:
1) fp32 parity — eager LoRA oracle (axis.lora on the tape engine) vs the
   compiled step: loss + EVERY adapter grad, bit-tight.
2) bf16 + flash parity — relaxed (bf16 storage).
3) 20 compiled LoRA steps — loss must fall (adapter-only AdamW works).
4) 1B-class LoRA fine-tune benchmark (bf16+flash): speed + memory vs the
   full-training numbers, plus a bigger batch that full training's optimizer
   state would not leave room for.

    modal run engine/modal_lora_test.py
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
app = modal.App("axis-engine-lora")

BV, BB, BT, BD, BH, BKV, BMLP, BL = 32000, 4, 2048, 1536, 24, 8, 4096, 48


def _compile_runtime():
    import subprocess
    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "--shared", "-Xcompiler", "-fPIC",
         "/root/runtime.cu", "-lcublas", "-o", "/root/libaxeng.so"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE FAILED:\n", r.stderr[-3000:], flush=True)
        raise SystemExit(1)
    print("nvcc: OK", flush=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def parity():
    import sys
    import numpy as np
    sys.path.insert(0, "/root")
    _compile_runtime()

    import axis
    from axis import nn, lora
    from axis.tensor import Tensor
    from axis.compile import compile_model

    V, B, T, D, H, KV, MLP, L = 500, 2, 32, 128, 4, 2, 256, 3
    rng = np.random.default_rng(0)
    toks = rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]

    def build():
        axis.manual_seed(0)
        m = nn.Transformer(vocab_size=V, dim=D, n_layers=L, n_heads=H,
                           n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=T,
                           tie_embeddings=True)
        lora.apply_lora(m, rank=4, alpha=16)      # all 7 targets (default)
        return m

    # eager oracle
    model = build()
    loss_t = model.loss(Tensor(inp), Tensor(tgt))
    ref_loss = float(loss_t.data)
    loss_t.backward()
    ref_g = {n: p.grad.copy() for n, p in model.named_parameters()
             if p.requires_grad}
    n_tr = sum(v.size for v in ref_g.values())
    n_all = sum(p.data.size for _, p in model.named_parameters())
    print(f"trainable: {n_tr:,} of {n_all:,} params "
          f"({100*n_tr/n_all:.2f}%)", flush=True)

    def check(label, dtype, impl, loss_tol, grad_tol):
        m2 = build()                                 # fresh identical model
        ct = compile_model(m2, B, T, lib_path="/root/libaxeng.so",
                           tf32=False, dtype=dtype, attn_impl=impl,
                           attn_tile=8)
        assert set(ct.pnames) == set(ref_g), "trainable-set mismatch"
        got_loss = ct.step(inp, tgt, t=1)
        got = ct.grads()
        worst = ("", 0.0)
        for nm, g in ref_g.items():
            rel = np.abs(got[nm] - g).max() / (np.abs(g).max() + 1e-9)
            if rel > worst[1]:
                worst = (nm, rel)
        ok = abs(ref_loss - got_loss) < loss_tol and worst[1] < grad_tol
        print(f"{label}: loss Δ {abs(ref_loss-got_loss):.2e} | "
              f"worst adapter grad rel {worst[1]:.2e} ({worst[0]}) "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    ok = True
    ok &= check("LoRA fp32 tiled", "fp32", "auto", 1e-4, 1e-3)
    ok &= check("LoRA bf16 flash", "bf16", "flash", 5e-2, 1.5e-1)
    ok &= check("LoRA bf16 tiled", "bf16", "tiled", 5e-2, 1.5e-1)

    # 20 compiled steps: loss falls (adapter-only AdamW; B zero-init means
    # step-0 loss == base-model loss)
    m3 = build()
    ct = compile_model(m3, B, T, lib_path="/root/libaxeng.so",
                       lr=1e-2, dtype="bf16", attn_impl="flash")
    losses = [ct.step(inp, tgt) for _ in range(20)]
    fell = losses[-1] < losses[0] - 0.2   # adapter-only training is gradual
    print(f"20 LoRA steps: {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"-> {'PASS' if fell else 'FAIL'}", flush=True)
    ok &= fell
    print(f"\nCOMPILED LoRA: {'PASS' if ok else 'FAIL'}", flush=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600, memory=49152,
              single_use_containers=True)
def bench(bsz: int = 4):
    import subprocess, sys, time
    import numpy as np
    sys.path.insert(0, "/root")
    _compile_runtime()
    from axis.compile import CompiledTransformer
    N = bsz * BT
    DH = BD // BH
    rng = np.random.default_rng(1)
    toks = rng.integers(0, BV, size=(bsz, BT + 1)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]

    shapes = {"embed.weight": (BV, BD), "norm.weight": (BD,)}
    for i in range(BL):
        p = f"blocks.{i}."
        shapes.update({p + "attn_norm.weight": (BD,), p + "mlp_norm.weight": (BD,),
                       p + "attn.q_proj.weight": (BD, BH * DH),
                       p + "attn.k_proj.weight": (BD, BKV * DH),
                       p + "attn.v_proj.weight": (BD, BKV * DH),
                       p + "attn.o_proj.weight": (BH * DH, BD),
                       p + "mlp.gate_proj.weight": (BD, BMLP),
                       p + "mlp.up_proj.weight": (BD, BMLP),
                       p + "mlp.down_proj.weight": (BMLP, BD)})
    r2 = np.random.default_rng(2)
    w = {nm: (np.ones(sh, dtype=np.float32) if "norm" in nm else
              (r2.standard_normal(sh) * 0.02).astype(np.float32))
         for nm, sh in shapes.items()}
    cfg = dict(vocab_size=BV, dim=BD, n_layers=BL, n_heads=BH, n_kv_heads=BKV,
               mlp_hidden=BMLP, tie_embeddings=True)
    ct = CompiledTransformer("/root/libaxeng.so", cfg, w, bsz, BT,
                             tf32=True, dtype="bf16", attn_impl="flash",
                             lora_r=16, lora_alpha=32)
    n_tr = sum(int(np.prod(s)) for s in ct.shapes.values())
    print(f"LoRA r=16: {n_tr/1e6:.1f}M trainable", flush=True)
    losses = [round(ct.step(inp, tgt), 3) for _ in range(2)]
    ct.capture()
    ct.replay_step(inp, tgt)
    t0 = time.perf_counter()
    for _ in range(5):
        ct.replay_step(inp, tgt)
    ms = (time.perf_counter() - t0) / 5 * 1000
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    mem = int(out.strip().splitlines()[0])
    label = f"axis LoRA bf16 flash seq{BT} b{bsz}"
    print(f"{label}: {ms:7.0f} ms/step | {N/(ms/1000):7.0f} tok/s "
          f"| mem {mem} MiB | first losses {losses}", flush=True)
    return (label, ms, N / (ms / 1000), mem)


@app.local_entrypoint()
def main():
    import os
    parity.remote()
    if os.environ.get("AXIS_PARITY_ONLY"):
        return
    rows = [bench.remote(b) for b in (4, 16)]
    print("\n== 1B-class LoRA fine-tune (r=16), seq 2048 ==")
    for name, ms, tps, mem in rows:
        print(f"{name:32s} {ms:7.0f} ms/step  {tps:8,.0f} tok/s  {mem:6d} MiB")
