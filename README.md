# Axis

**India's first proprietary AI training & fine-tuning framework — by [Zyora Labs](https://zyoralabs.com).**

Axis trains and fine-tunes transformer models on the GPU through a compiled
C++/CUDA engine: the entire training step — forward, backward, and the AdamW
update — is lowered into a single execution plan and run as one native call,
or captured as a CUDA graph. Every result is **proven numerically identical to
a reference implementation** before it ships.

Made in India. Proprietary framework, © Zyora Labs.

## Highlights

- **Compiled training engine.** The whole step (embed → N transformer blocks →
  norm → head → cross-entropy → full backward → AdamW) is lowered to one
  execution plan and executed natively — or captured as a CUDA graph and
  replayed.
- **bf16 with fp32 masters.** bf16 storage + tensor-core GEMMs, fp32 master
  weights and fp32 accumulation — no loss scaling needed.
- **Fused flash attention (forward *and* backward).** WMMA tensor-core kernels;
  attention probabilities never touch memory (recomputed on-chip from the saved
  log-sum-exp), so memory stays flat as sequence length grows.
- **Multi-GPU.** NCCL data parallelism at **99% weak-scaling efficiency**, plus
  **ZeRO-1 optimizer-state sharding** to train models whose optimizer state is
  larger than one GPU can hold — bit-identical to replicated data parallel.
- **LoRA fine-tuning** on the compiled engine — the frozen base carries no
  optimizer state, so LoRA fine-tunes in **~38% less GPU memory** than
  HuggingFace + PEFT at comparable speed.
- **A complete training loop.** Learning-rate schedules (under CUDA graphs),
  global-norm gradient clipping, correct weight decay (1D params excluded),
  `ignore_index` / padding masking for real instruction data, and fully
  resumable checkpoints — all graph-safe.
- **Numerically exact.** The compiled engine matches its reference to **3.1e-07
  over 30 full training steps** — verified end to end through the optimizer.
  Nothing ships without a parity gate.

## Install

```bash
pip install -e .

# Build the CUDA runtime (once, on a machine with nvcc):
nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC \
     engine/runtime.cu -lcublas -o libaxeng.so
```

## Quickstart — train on the compiled engine

```python
import numpy as np
import axis
from axis import nn

axis.manual_seed(42)

# A Llama-class decoder: RoPE + RMSNorm + SwiGLU + grouped-query attention
model = nn.Transformer(
    vocab_size=32000, dim=1536, n_layers=48,
    n_heads=24, n_kv_heads=8, mlp_hidden=4096, max_seq_len=2048,
)

# Compile the whole training step to the CUDA engine (bf16 + fused flash attn)
ct = axis.compile_model(model, batch=4, seq=2048, dtype="bf16",
                        lr=3e-4, wd=0.1, max_grad_norm=1.0)

# One native call per step = forward + backward + AdamW
for x, y in loader:                     # x, y : int64 [batch, seq]
    loss = ct.step(x, y)

# Learning-rate schedule (works under CUDA graph replay)
ct.set_lr(new_lr)

# Resumable checkpoint (weights + optimizer moments + step + lr)
state = ct.state_dict()
ct.load_state_dict(state)
```

Capture the step as a CUDA graph for the lowest per-step overhead:

```python
ct.capture()
for x, y in loader:
    loss = ct.replay_step(x, y)
```

## Fine-tune a pretrained model with LoRA

```python
import axis
from axis import lora, HFTokenizer

# 1. Load a real HuggingFace Llama-family checkpoint (config.json + safetensors)
model = axis.from_pretrained("path/to/llama-model")
tok   = HFTokenizer.from_pretrained("path/to/llama-model")

# 2. Inject LoRA adapters (freezes the base, trains ~0.5% of params)
lora.apply_lora(model, rank=16, alpha=32)

# 3. Compile — LoRA is detected automatically; only the adapters train
ct = axis.compile_model(model, batch=4, seq=2048, dtype="bf16", lr=1e-4)

# Real instruction data: set masked / padding positions in y to -1 (ignored)
for x, y in loader:
    loss = ct.step(x, y)
```

## Multi-GPU

Data parallelism (one process per GPU), with the gradient all-reduce fused
into the plan between backward and the optimizer step:

```python
ct = axis.compile_model(model, batch=4, seq=2048, dtype="bf16",
                        grad_sync=True, rank=rank, world=world)
ct.eng.nccl_init(rank, world, uid)      # rank-0 shares the ncclUniqueId
```

Add ZeRO-1 to shard the optimizer state across ranks (train models whose
optimizer state exceeds one GPU):

```python
ct = axis.compile_model(model, batch=4, seq=2048, dtype="bf16",
                        grad_sync=True, zero_stage=1, rank=rank, world=world)
```

Build the runtime with NCCL for multi-GPU:

```bash
nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC -DAXIS_NCCL \
     -I$NCCL/include engine/runtime.cu -L$NCCL/lib -lnccl -lcublas -o libaxeng.so
```

## Benchmarks

Honest head-to-head against the real PyTorch stack (HuggingFace Transformers +
SDPA flash attention, PEFT), same model and shape, single **NVIDIA A100-80GB**.
Reproducible: `modal run engine/modal_compare_test.py`.

**1B-class** (dim 1536, 48 layers, 24h/8kv, seq 2048, batch 4, bf16):

*Training*

| | ms/step | tok/s | memory |
|---|---:|---:|---:|
| PyTorch (HF + SDPA) | 574 | 14,280 | 36.8 GB |
| PyTorch (`torch.compile`) | 641 | 12,783 | 35.2 GB |
| **Axis** | 797 | 10,279 | 42.1 GB |

*Fine-tuning (LoRA r=16)*

| | ms/step | tok/s | memory |
|---|---:|---:|---:|
| PyTorch + PEFT | 677 | 12,096 | 36.6 GB |
| **Axis** | 701 | 11,689 | **22.8 GB** |

At 1B training PyTorch is faster and leaner — a decade of tuning behind it.
Axis's clear win is **LoRA fine-tuning memory: 38% less than HF + PEFT** at
comparable speed (the frozen base carries no optimizer state and stays in bf16).

**Multi-GPU scaling** (Axis, 1B-class bf16 + flash, batch 4 / GPU):

| GPUs | tok/s | efficiency |
|---|---:|---:|
| 1× A100 | 10,036 | — |
| 2× A100 | 19,950 | **99%** |

## Correctness

Every capability is gated against a reference implementation before it ships:

- Compiled fp32 vs reference, 30 full training steps: **max rel drift 3.1e-07**.
- 32/32 shape-fuzz configs (varied depth/width/heads/kv/vocab/tied/LoRA).
- 500-step CUDA-graph soak: converges, zero NaN, exact bias correction.
- LoRA adapter gradients bit-tight; multi-GPU + ZeRO-1 bit-identical to
  replicated data parallel.
- Loads real HuggingFace Llama checkpoints and reproduces logits to machine
  precision (max |Δ| ≈ 3e-6).

## What's inside

| Module | Contents |
|---|---|
| `axis.compile` | `compile_model` / `CompiledTransformer` — lowers the full training step to the CUDA engine; bf16, flash attention, CUDA graphs, LoRA, DDP, ZeRO-1 |
| `axis.engine` | ctypes binding to the C ABI runtime (execution plans, graph capture, NCCL) |
| `engine/runtime.cu` | C++/CUDA runtime — own cuBLAS handle, fused kernels, WMMA flash attention (fwd+bwd), AdamW, NCCL collectives |
| `axis.nn` | `Transformer`, RoPE, RMSNorm, SwiGLU, grouped-query attention |
| `axis.pretrained` | `from_pretrained` — HuggingFace safetensors loader |
| `axis.tokenizer` | `HFTokenizer` — byte-level BPE (GPT-2 / Llama-3 / Qwen / Mistral) |
| `axis.lora` | LoRA adapters, freeze / merge, adapter-only state dict |
| `axis.optim` | AdamW, global-norm grad clipping, cosine warmup |
| `axis.generate` | Autoregressive generation (greedy / temperature / top-k) |
| `axis.data` | `LMDataset`, `DataLoader` |

## Tests

```bash
pip install -e ".[dev,hf]"
pytest tests/ -q          # 103 reference tests: gradchecks, determinism, LoRA, tokenizer, GQA
```

The compiled engine is validated on GPU (A100) — parity harnesses live under
`engine/modal_*.py`.

## About

**Axis** is a proprietary framework by **[Zyora Labs](https://zyoralabs.com)** —
India's first proprietary AI training & fine-tuning framework.

- **Developer / Founder:** Vasanth
- **Email:** [vasu@zyoralabs.com](mailto:vasu@zyoralabs.com)
- **Website:** [zyoralabs.com](https://zyoralabs.com)

© Zyora Labs. All rights reserved.
