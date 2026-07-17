"""axis.optim — optimizers + LR schedules for transformer training.

AdamW is the reference optimizer (decoupled weight decay, exactly the
Loshchilov-Hutter formulation PyTorch uses). Gradient clipping is global-norm,
matching torch.nn.utils.clip_grad_norm_.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from axis.nn import Parameter


class AdamW:
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self._m[i] = b1 * self._m[i] + (1.0 - b1) * g
            self._v[i] = b2 * self._v[i] + (1.0 - b2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            # Decoupled weight decay (applied directly to weights).
            if self.weight_decay > 0.0:
                p.data *= (1.0 - self.lr * self.weight_decay)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    # ── checkpoint support ──
    def state_dict(self) -> dict:
        return {
            "t": self.t,
            "m": [m.copy() for m in self._m],
            "v": [v.copy() for v in self._v],
            "lr": self.lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.t = int(state["t"])
        self._m = [np.asarray(m) for m in state["m"]]
        self._v = [np.asarray(v) for v in state["v"]]
        self.lr = float(state.get("lr", self.lr))


class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float = 0.01, momentum: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self._buf = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.momentum > 0.0:
                self._buf[i] = self.momentum * self._buf[i] + p.grad
                p.data -= self.lr * self._buf[i]
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def clip_grad_norm(params: Iterable[Parameter], max_norm: float) -> float:
    """Global-norm gradient clipping. Returns the pre-clip norm."""
    params = [p for p in params if p.grad is not None]
    total = math.sqrt(sum(float((p.grad * p.grad).sum()) for p in params))
    if total > max_norm and total > 0.0:
        scale = max_norm / total
        for p in params:
            p.grad *= scale
    return total


class CosineWithWarmup:
    """Linear warmup → cosine decay to min_lr. Call .step() once per optimizer step."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 max_lr: float, min_lr: float = 0.0):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0

    def get_lr(self) -> float:
        s = self.step_num
        if s < self.warmup:
            return self.max_lr * (s + 1) / self.warmup
        if s >= self.max_steps:
            return self.min_lr
        progress = (s - self.warmup) / max(1, self.max_steps - self.warmup)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        lr = self.get_lr()
        self.opt.lr = lr
        self.step_num += 1
        return lr
