"""Head-to-head at 7B-class scale: Axis vs the real PyTorch stack, same shape
(Llama-2-7B: dim4096 / 32L / 32h·32kv / mlp11008 / V32000, tied embeds), one
A100-80GB, per-config fresh container (clean GPU memory).

Full-model 7B training does NOT fit on a single 80GB GPU for EITHER framework
(bf16 weights + grads + fp32 AdamW master/m/v ≈ 112 GB), so the single-GPU 7B
benchmark is LoRA fine-tuning — which is also Axis's genuine strength.

Fine-tuning (LoRA r=16, q/k/v/o/gate/up/down), seq 2048, batch 1, bf16:
  - axis-lora       : Axis compiled LoRA bf16 + fused flash + CUDA graph
  - torch-peft-lora : HF Llama + PEFT LoRA, bf16 autocast, SDPA (flash) attention

    modal run engine/modal_7b_bench.py
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
app = modal.App("axis-7b-bench")

# Llama-2-7B shape (tied embeddings for a single shared weight schema).
V, B, T, D, H, KV, MLP, L = 32000, 1, 2048, 4096, 32, 32, 11008, 32
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

    if mode == "axis-lora":
        _axis_runtime()
        from axis.compile import CompiledTransformer
        cfg = dict(vocab_size=V, dim=D, n_layers=L, n_heads=H, n_kv_heads=KV,
                   mlp_hidden=MLP, tie_embeddings=True)
        ct = CompiledTransformer("/root/libaxeng.so", cfg, _axis_weights(), B, T,
                                 tf32=True, dtype="bf16", attn_impl="flash",
                                 lora_r=16, lora_alpha=32)
        ct.step(inp, tgt); ct.step(inp, tgt)
        ct.capture(); ct.replay_step(inp, tgt)
        t0 = time.perf_counter()
        for _ in range(NSTEP):
            ct.replay_step(inp, tgt)
        ms = (time.perf_counter() - t0) / NSTEP * 1000
        n_tr = 2 * 16 * (H * (D // H) + KV * (D // H) + KV * (D // H)
                         + H * (D // H) + MLP + MLP + MLP) * L  # rough r*(fin+fout)
        return (f"axis-lora ({n_tr/1e6:.0f}M trainable)", ms, N / (ms / 1000), _mem_mb())

    # ---- torch stack ----
    import torch
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    m = _llama()
    ti = torch.tensor(inp).cuda(); tt = torch.tensor(tgt).cuda()

    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=16, lora_alpha=32, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0, bias="none")
    m = get_peft_model(m, lc)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(input_ids=ti, labels=None)
            loss = F.cross_entropy(out.logits.float().reshape(-1, V), tt.reshape(-1))
        loss.backward()
        opt.step()

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NSTEP):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / NSTEP * 1000
    return (f"torch-peft-lora ({n_tr/1e6:.0f}M trainable)", ms, N / (ms / 1000), _mem_mb())


@app.local_entrypoint()
def main():
    print("== 7B-class FINE-TUNE (Llama-2-7B shape, LoRA r=16, seq2048, batch1, bf16, A100-80GB) ==", flush=True)
    res = [bench.remote(m) for m in ("axis-lora", "torch-peft-lora")]
    for name, ms, tps, mem in res:
        print(f"{name:34s} {ms:8.0f} ms/step  {tps:9,.0f} tok/s  {mem:6d} MiB  ({mem/1024:.1f} GB)", flush=True)
