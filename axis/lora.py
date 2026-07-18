"""axis.lora — Low-Rank Adaptation for parameter-efficient fine-tuning.

LoRA freezes the pretrained weights and trains a tiny low-rank update
W_eff = W + (alpha/r) * A @ B, where A is [in, r] and B is [r, out] (Axis's
Linear weight is [in, out]). B starts at zero, so the adapted model is
identical to the base model at step 0 — fine-tuning only nudges it.

Typical use:
    model = axis.from_pretrained("...")
    axis.lora.apply_lora(model, rank=8, alpha=16)          # freeze + inject
    opt = optim.AdamW(axis.lora.trainable_parameters(model), lr=1e-4)
    # ... train ...
    axis.lora.merge_lora(model)                            # fold in for inference
"""
from __future__ import annotations

from typing import Iterable, Iterator, List

import numpy as np

from axis import ops
from axis.nn import Linear, Module, Parameter
from axis.tensor import Tensor, get_rng

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


class LoRALinear(Module):
    """Wraps a frozen Linear with a trainable low-rank update."""

    def __init__(self, base: Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if base.bias is not None:
            base.bias.requires_grad = False
        base.weight.requires_grad = False
        self.base = base                       # frozen
        in_f, out_f = base.weight.shape        # Axis Linear weight is [in, out]
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        # A ~ small random, B = 0  =>  initial update is exactly zero.
        a = get_rng().standard_normal((in_f, rank)).astype(np.float32) / np.sqrt(rank)
        self.lora_a = Parameter(a)
        self.lora_b = Parameter(np.zeros((rank, out_f), dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        delta = ops.matmul(ops.matmul(x, self.lora_a), self.lora_b)
        return ops.add(base_out, ops.mul(delta, Tensor(np.float32(self.scaling))))

    def merged_weight(self) -> np.ndarray:
        """Base weight with the LoRA update folded in ([in, out])."""
        return (self.base.weight.data
                + self.scaling * (self.lora_a.data @ self.lora_b.data)).astype(np.float32)


def _all_modules(root: Module) -> Iterator[Module]:
    """Yield every Module in the tree (handles ModuleList children)."""
    seen = []
    stack = [root]
    while stack:
        m = stack.pop()
        if id(m) in [id(s) for s in seen]:
            continue
        seen.append(m)
        yield m
        for child in getattr(m, "_modules", {}).values():
            if isinstance(child, Module):
                stack.append(child)
            else:  # ModuleList
                for item in child:
                    stack.append(item)


def freeze(model: Module) -> None:
    """Freeze every parameter (requires_grad = False)."""
    for p in model.parameters():
        p.requires_grad = False


def trainable_parameters(model: Module) -> List[Parameter]:
    """Parameters that still require grad — i.e. the LoRA adapters."""
    return [p for p in model.parameters() if p.requires_grad]


def apply_lora(model: Module, rank: int = 8, alpha: float = 16.0,
               target_modules: Iterable[str] = DEFAULT_TARGETS) -> Module:
    """Freeze the whole model and replace the named Linear layers with
    LoRALinear adapters. Returns the model (mutated in place)."""
    freeze(model)
    targets = set(target_modules)
    for m in list(_all_modules(model)):
        for name, child in list(getattr(m, "_modules", {}).items()):
            if isinstance(child, Linear) and name in targets:
                setattr(m, name, LoRALinear(child, rank=rank, alpha=alpha))
    return model


def merge_lora(model: Module) -> Module:
    """Fold every LoRA update back into its base Linear weight and swap the
    plain (now unfrozen-shape) Linear back in — for fast inference / export."""
    for m in list(_all_modules(model)):
        for name, child in list(getattr(m, "_modules", {}).items()):
            if isinstance(child, LoRALinear):
                base = child.base
                base.weight.data = child.merged_weight()
                setattr(m, name, base)
    return model


def lora_state_dict(model: Module) -> dict:
    """Just the adapter weights — small enough to save/share on their own."""
    return {name: p.data.copy() for name, p in model.named_parameters()
            if "lora_a" in name or "lora_b" in name}
