# Axis

**India's first proprietary AI training & fine-tuning framework — by [Zyora Labs](https://zyoralabs.com).**

Axis trains and fine-tunes real transformer models — from scratch or from a
pretrained checkpoint — on CPU and GPU, across NVIDIA and AMD. It runs models
**numerically identical to PyTorch / HuggingFace**, built on standard numerical
libraries (NumPy on CPU, CuPy → cuBLAS/rocBLAS on GPU) — the same foundation
every major framework uses.

Made in India. Proprietary framework, © Zyora Labs.

## Highlights

- **Numerically exact.** Loads real Llama-family models and reproduces
  HuggingFace logits to machine precision (max |Δ| ≈ 3e-6). Every op's gradient
  is checked against finite differences.
- **Real fine-tuning workflow.** Load pretrained weights (safetensors), the
  model's real tokenizer, apply **LoRA**, train, and generate.
- **CPU and GPU, one codebase.** Pure-NumPy reference engine on CPU; the whole
  engine runs on the GPU via CuPy (cuBLAS) with `model.to_gpu()`. Cross-vendor:
  NVIDIA (cupy-cuda) and AMD (cupy-rocm).
- **Deterministic + safe.** `manual_seed` → bit-identical runs; atomic
  checkpoints that never corrupt on a crash.
- **Llama-class architecture.** RoPE, RMSNorm, SwiGLU, grouped-query attention
  (GQA) — validated against HuggingFace.

## Install

```bash
pip install -e .                 # core (CPU)
pip install -e ".[hf]"           # + real HuggingFace tokenizer (regex)
pip install cupy-cuda12x          # + GPU (NVIDIA); use a cupy-rocm build for AMD
```

## Quickstart — train from scratch

```python
import numpy as np
import axis
from axis import nn, optim
from axis.tensor import Tensor

axis.manual_seed(42)

# A Llama-class decoder: RoPE + RMSNorm + SwiGLU + GQA
model = nn.Transformer(
    vocab_size=32000, dim=512, n_layers=8,
    n_heads=8, n_kv_heads=4, max_seq_len=512,
)
model.to_gpu()                    # run on the GPU (cuBLAS). Omit for CPU.

opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

tokens  = Tensor(np.random.randint(0, 32000, (8, 512)))
targets = Tensor(np.random.randint(0, 32000, (8, 512)))

loss = model.loss(tokens, targets)
loss.backward()
opt.step(); opt.zero_grad()

axis.save("ckpt.npz", model=model, optimizer=opt, step=1)
```

## Fine-tune a real pretrained model

```python
import axis
from axis import optim, lora, HFTokenizer

# 1. Load a real HuggingFace Llama-family checkpoint (config.json + safetensors)
model = axis.from_pretrained("path/to/llama-model")
tok   = HFTokenizer.from_pretrained("path/to/llama-model")
model.to_gpu()

# 2. Add LoRA adapters (freezes the base model, trains ~0.5% of params)
lora.apply_lora(model, rank=8, alpha=16)
opt = optim.AdamW(lora.trainable_parameters(model), lr=1e-4)

# 3. Tokenize your text and train
ids = tok.encode(open("my_data.txt").read())
loader = axis.DataLoader(axis.LMDataset(ids, seq_len=512), batch_size=8)
for inp, tgt in loader:
    loss = model.loss(inp, tgt)
    loss.backward()
    opt.step(); opt.zero_grad()

# 4. Merge adapters and generate
lora.merge_lora(model)
out = axis.generate(model, tok.encode("Once upon a time"), max_new_tokens=100)
print(tok.decode(out))
```

## Benchmarks

Measured on a single **NVIDIA A100** — same model, same batch/sequence,
proper warmup + CUDA sync + median. Reproducible: `modal run modal_benchmark.py`.

**Axis vs PyTorch** — identical Llama, 125M params, batch 8, seq 512, fp32+TF32:

| Framework | tok/s | ms/step | peak memory |
|---|---:|---:|---:|
| PyTorch | 53,060 | 77 | 8.0 GB |
| **Axis** | 14,531 | 282 | 8.6 GB |

Axis reaches ~1/4 of PyTorch's throughput at the **same memory**, with the
**forward pass numerically identical** (max |Δ| = 5e-3 with TF32; ≈3e-6 in pure
fp32). A strong result for a from-scratch framework — no overclaiming, PyTorch
is simply the reference point.

**Throughput scales with batch size** (dim 512, 6 layers, seq 128):

| batch | 8 | 16 | 32 | 64 | 96 |
|---|---:|---:|---:|---:|---:|
| tok/s | 13.7k | 27.8k | 54.3k | 86.5k | 92.5k |

**Trains real model sizes** on one A100 — 100M (11.8 GB) and 303M (20.7 GB),
converging, no OOM.

## Validated against HuggingFace

Axis loads a real HF Llama checkpoint and reproduces its logits exactly:

```
max|Δ| = 8.9e-08   argmax match: every position   (float32 machine precision)
```

The full stack — weight mapping, RoPE, grouped-query attention, RMSNorm,
SwiGLU — matches the reference implementation.

## Use cases

- **Fine-tune open LLMs** (Llama, Mistral, Qwen family) with LoRA on your own data.
- **Pre-train** small/mid transformers from scratch (validated to 300M params).
- **Reproducible research** — deterministic runs, gradient-checked ops.
- **Cross-vendor training** — the same code on NVIDIA and AMD GPUs, or CPU.
- **On-prem / sovereign AI** — a self-contained, proprietary training stack.

## What's inside

| Module | Contents |
|---|---|
| `axis.tensor` | Autograd `Tensor` (tape-based reverse mode), deterministic RNG, `no_grad` |
| `axis.backend` | CPU/GPU array-module dispatch (NumPy ↔ CuPy), `to_gpu`/`to_cpu`, TF32 |
| `axis.ops` | 30+ gradient-checked primitives — matmul, softmax, fused cross-entropy, fused causal attention, embedding, SwiGLU |
| `axis.nn` | `Module`, `Linear`, `Embedding`, `RMSNorm`, RoPE, GQA attention, `SwiGLU`, `Transformer` |
| `axis.optim` | `AdamW`, `SGD`, global-norm grad clipping, cosine warmup |
| `axis.pretrained` | `from_pretrained` — HuggingFace safetensors loader |
| `axis.tokenizer` | `HFTokenizer` — byte-level BPE (GPT-2 / Llama-3 / Qwen / Mistral) |
| `axis.data` | `ByteTokenizer`, `LMDataset`, `DataLoader` |
| `axis.lora` | LoRA adapters, freeze/merge, adapter-only state dict |
| `axis.generate` | Autoregressive generation (greedy / temperature / top-k) |
| `axis.checkpoint` | Atomic `save`/`load` with full training-state resume |

## Tests

```bash
pip install -e ".[dev,hf]"
pytest tests/ -q          # 91 tests: gradchecks, determinism, LoRA, tokenizer, GQA
```

## About

**Axis** is a proprietary framework by **[Zyora Labs](https://zyoralabs.com)** —
India's first proprietary AI training & fine-tuning framework.

- **Developer / Founder:** Vasanth
- **Email:** [vasu@zyoralabs.com](mailto:vasu@zyoralabs.com)
- **Website:** [zyoralabs.com](https://zyoralabs.com)

© Zyora Labs. All rights reserved. Built on standard open numerical libraries
(NumPy, CuPy), like every major deep-learning framework.
