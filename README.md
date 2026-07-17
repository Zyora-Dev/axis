# Axis

**A reliable framework for training and fine-tuning transformers — by Zyora Labs.**

Axis is built for one job: training real AI models — transformer pre-training
from scratch and fine-tuning — with reliability as the first-class feature.

## Why Axis

- **Every gradient is verified.** Every op's analytic backward is checked
  against central finite differences (`axis.gradcheck`). No silent wrong
  gradients — the #1 killer of training runs.
- **Deterministic by default.** `axis.manual_seed(n)` makes two runs
  bit-identical. Debugging a training run should never involve luck.
- **Atomic checkpoints.** Save is write-temp → fsync → atomic rename. A crash
  mid-save can never corrupt your last good checkpoint. Optimizer state, LR
  schedule and metadata all resume exactly.
- **A NumPy reference engine as ground truth.** Phase 2 adds a GPU backend via
  [locomp](https://github.com/Zyora-Dev/locomp) (Metal · CUDA · ROCm · RISC-V)
  that must match this reference bit-for-bit.

## Quickstart

```python
import numpy as np
import axis
from axis import nn, optim
from axis.tensor import Tensor

axis.manual_seed(42)

# A Llama-class decoder: RoPE + RMSNorm + SwiGLU + GQA
model = axis.nn.Transformer(
    vocab_size=32000, dim=256, n_layers=8,
    n_heads=8, n_kv_heads=4, max_seq_len=512,
)
opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
sched = optim.CosineWithWarmup(opt, warmup_steps=100, max_steps=10_000, max_lr=3e-4)

tokens  = Tensor(np.random.randint(0, 32000, (2, 128)))
targets = Tensor(np.random.randint(0, 32000, (2, 128)))

model.zero_grad()
loss = model.loss(tokens, targets)
loss.backward()
optim.clip_grad_norm(model.parameters(), 1.0)
sched.step()
opt.step()

axis.save("ckpt.npz", model=model, optimizer=opt, scheduler=sched, step=1)
```

## What's inside

| Module | Contents |
|---|---|
| `axis.tensor` | Autograd `Tensor` (tape-based reverse mode), deterministic RNG, `no_grad` |
| `axis.ops` | 30+ differentiable primitives — matmul (batched+broadcast), softmax, cross-entropy (fused, ignore_index), embedding, GELU/SiLU, reductions, shape ops |
| `axis.nn` | `Module`/`Parameter`, `Linear`, `Embedding`, `RMSNorm`, `LayerNorm`, RoPE, causal GQA attention, `SwiGLU`, `TransformerBlock`, `Transformer` |
| `axis.optim` | `AdamW` (decoupled weight decay), `SGD`, global-norm grad clipping, cosine warmup schedule |
| `axis.checkpoint` | Atomic `save`/`load` with full training-state resume |
| `axis.gradcheck` | Numerical gradient verification for any scalar function |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

The suite gradchecks **every op**, verifies determinism (bit-identical runs),
round-trips checkpoints, and proves end-to-end correctness by training a small
transformer until it memorizes a copy task (loss < 0.1).

## Roadmap

- **Phase 1 (this):** NumPy reference engine — full autograd, transformer
  blocks, AdamW, checkpoints. The ground truth.
- **Phase 2:** locomp GPU backend (Metal / CUDA / ROCm / RISC-V) validated
  against Phase 1 outputs.
- **Phase 3:** LoRA/QLoRA fine-tuning, bf16 mixed precision, gradient
  checkpointing, HF weight import.
- **Phase 4:** multi-GPU data parallel.

## License

Apache-2.0
