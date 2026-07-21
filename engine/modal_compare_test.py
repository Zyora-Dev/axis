"""Head-to-head: Axis vs the real-world PyTorch stack, same 1B-class shape
(dim1536 / 48L / 24h·8kv / mlp4096 / seq2048 / batch4 / bf16 / V32000) on one
A100-80GB. Each config runs in its own fresh container (clean GPU memory).

Training:
  - axis            : Axis compiled bf16 + fused flash + CUDA graph
  - torch-eager     : HF LlamaForCausalLM, bf16 autocast, SDPA (flash) attention
  - torch-compile   : same + torch.compile(mode="max-autotune")
Fine-tuning (LoRA r=16, q/k/v/o/gate/up/down):
  - axis-lora       : Axis compiled LoRA bf16 + flash
  - torch-peft-lora : HF Llama + PEFT LoRA, bf16 autocast, SDPA

    modal run engine/modal_compare_test.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "torch", "transformers", "peft", "accelerate")
    .add_local_file(str(REPO / "axis" / "_csrc" / "runtime.cu"), remote_path="/root/runtime.cu")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-compare")

V, B, T, D, H, KV, MLP, L = 32000, 4, 2048, 1536, 24, 8, 4096, 48
N = B * T
NSTEP = 5


def _mem_mb():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def _llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=V, hidden_size=D, intermediate_size=MLP,
                      num_hidden_layers=L, num_attention_heads=H,
                      num_key_value_heads=KV, max_position_embeddings=T,
                      rms_norm_eps=1e-5, tie_word_embeddings=True,
                      attn_implementation="sdpa")
    return LlamaForCausalLM(cfg).cuda().to(torch.bfloat16)


def _axis_runtime():
    import subprocess
    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "--shared", "-Xcompiler", "-fPIC",
         "/root/runtime.cu", "-lcublas", "-o", "/root/libaxeng.so"],
        capture_output=True, text=True)
    if r.returncode:
        print("nvcc FAIL\n", r.stderr[-2000:], flush=True); raise SystemExit(1)


def _axis_weights(seed=2):
    import numpy as np
    DH = D // H
    shapes = {"embed.weight": (V, D), "norm.weight": (D,)}
    for i in range(L):
        p = f"blocks.{i}."
        shapes.update({p + "attn_norm.weight": (D,), p + "mlp_norm.weight": (D,),
                       p + "attn.q_proj.weight": (D, H * DH),
                       p + "attn.k_proj.weight": (D, KV * DH),
                       p + "attn.v_proj.weight": (D, KV * DH),
                       p + "attn.o_proj.weight": (H * DH, D),
                       p + "mlp.gate_proj.weight": (D, MLP),
                       p + "mlp.up_proj.weight": (D, MLP),
                       p + "mlp.down_proj.weight": (MLP, D)})
    r = np.random.default_rng(seed)
    return {nm: (np.ones(sh, np.float32) if "norm" in nm else
                 (r.standard_normal(sh) * 0.02).astype(np.float32)) for nm, sh in shapes.items()}


@app.function(image=image, gpu="A100-80GB", timeout=3600, single_use_containers=True)
def bench(mode: str):
    import time
    import numpy as np
    import sys
    sys.path.insert(0, "/root")
    rng = np.random.default_rng(1)
    toks = rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
    inp, tgt = toks[:, :-1], toks[:, 1:]

    if mode.startswith("axis"):
        _axis_runtime()
        from axis.compile import CompiledTransformer
        cfg = dict(vocab_size=V, dim=D, n_layers=L, n_heads=H, n_kv_heads=KV,
                   mlp_hidden=MLP, tie_embeddings=True)
        kw = dict(tf32=True, dtype="bf16", attn_impl="flash")
        if mode == "axis-lora":
            kw.update(lora_r=16, lora_alpha=32)
        ct = CompiledTransformer("/root/libaxeng.so", cfg, _axis_weights(), B, T, **kw)
        ct.step(inp, tgt); ct.step(inp, tgt)
        ct.capture(); ct.replay_step(inp, tgt)
        t0 = time.perf_counter()
        for _ in range(NSTEP):
            ct.replay_step(inp, tgt)
        ms = (time.perf_counter() - t0) / NSTEP * 1000
        return (mode, ms, N / (ms / 1000), _mem_mb())

    # ---- torch stack ----
    import torch
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = True
    m = _llama()
    ti = torch.tensor(inp).cuda(); tt = torch.tensor(tgt).cuda()

    if mode == "torch-peft-lora":
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=16, lora_alpha=32, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0, bias="none")
        m = get_peft_model(m, lc)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    fwd = m
    if mode == "torch-compile":
        fwd = torch.compile(m, mode="max-autotune")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fwd(input_ids=ti, labels=None)
            logits = out.logits
            loss = F.cross_entropy(logits.float().reshape(-1, V), tt.reshape(-1))
        loss.backward()
        opt.step()

    warm = 5 if mode == "torch-compile" else 2   # compile needs warmup
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NSTEP):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / NSTEP * 1000
    tag = f"{mode} ({n_tr/1e6:.0f}M trainable)" if "lora" in mode else mode
    return (tag, ms, N / (ms / 1000), _mem_mb())


@app.local_entrypoint()
def main():
    import os
    if not os.environ.get("AXIS_LORA_ONLY"):
        print("== TRAINING (1B-class, seq2048, batch4, bf16) ==", flush=True)
        train = [bench.remote(m) for m in ("axis", "torch-eager", "torch-compile")]
        for name, ms, tps, mem in train:
            print(f"{name:22s} {ms:7.0f} ms/step  {tps:8,.0f} tok/s  {mem:6d} MiB", flush=True)
    print("\n== FINE-TUNE (LoRA r=16, q/k/v/o/gate/up/down) ==", flush=True)
    lora = [bench.remote(m) for m in ("axis-lora", "torch-peft-lora")]
    for name, ms, tps, mem in lora:
        print(f"{name:28s} {ms:7.0f} ms/step  {tps:8,.0f} tok/s  {mem:6d} MiB", flush=True)
