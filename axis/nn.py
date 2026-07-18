"""axis.nn — Module system + transformer building blocks.

Everything a decoder-only transformer needs, composed from gradcheck-verified
primitives: Linear, Embedding, RMSNorm, LayerNorm, RoPE, causal multi-head
attention with GQA, SwiGLU MLP, TransformerBlock, and Transformer (the full
Llama-class model).
"""
from __future__ import annotations

import math
from typing import Iterator, Optional

import numpy as np

from axis import ops
from axis.tensor import Tensor, get_rng, no_grad


# ─── Module / Parameter system ──────────────────────────────────────────────


class Parameter(Tensor):
    """A Tensor that is registered as trainable by Module."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """Base class. Auto-registers Parameters and sub-Modules by attribute."""

    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, ModuleList):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield (f"{prefix}{name}", p)
        for mname, m in self._modules.items():
            yield from m.named_parameters(prefix=f"{prefix}{mname}.")

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        if strict and (missing or unexpected):
            raise KeyError(f"load_state_dict: missing={missing} unexpected={unexpected}")
        for name, p in own.items():
            if name in state:
                arr = np.asarray(state[name], dtype=p.data.dtype)
                if arr.shape != p.data.shape:
                    raise ValueError(f"{name}: shape {arr.shape} != {p.data.shape}")
                p.data = arr.copy()

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self) -> "Module":
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self) -> "Module":
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def to_gpu(self) -> "Module":
        """Move every parameter to the GPU (CuPy). The whole engine then runs
        on the device — matmul is cuBLAS, elementwise ops run on the GPU, with
        no per-op host round-trip."""
        from axis import backend as _B
        if not _B.has_cupy():
            raise RuntimeError("CuPy is not installed — GPU engine unavailable "
                               "(pip install cupy-cuda12x)")
        for p in self.parameters():
            p.data = _B.to_gpu_array(p.data)
            p._dev = None
        _B.set_device("gpu")
        return self

    def to_cpu(self) -> "Module":
        """Move every parameter back to host (NumPy)."""
        from axis import backend as _B
        for p in self.parameters():
            p.data = _B.to_numpy(p.data)
            p._dev = None
        _B.set_device("cpu")
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModuleList:
    """A list of Modules that registers its children."""

    def __init__(self, modules):
        self._items = list(modules)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)

    def parameters(self):
        for m in self._items:
            yield from m.parameters()

    def named_parameters(self, prefix: str = ""):
        for i, m in enumerate(self._items):
            yield from m.named_parameters(prefix=f"{prefix}{i}.")

    def train(self):
        for m in self._items:
            m.train()

    def eval(self):
        for m in self._items:
            m.eval()


# ─── Layers ─────────────────────────────────────────────────────────────────


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Kaiming-uniform-ish init, deterministic via axis RNG.
        bound = 1.0 / math.sqrt(in_features)
        w = (get_rng().uniform(-bound, bound, (in_features, out_features))).astype(np.float32)
        self.weight = Parameter(w)
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = ops.matmul(x, self.weight)
        if self.bias is not None:
            out = ops.add(out, self.bias)
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        w = (get_rng().standard_normal((num_embeddings, dim)) * 0.02).astype(np.float32)
        self.weight = Parameter(w)

    def forward(self, indices: Tensor) -> Tensor:
        return ops.embedding(self.weight, indices)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        ms = ops.mean(ops.mul(x, x), axis=-1, keepdims=True)
        inv = ops.pow(ops.add(ms, Tensor(np.float32(self.eps))), -0.5)
        return ops.mul(ops.mul(x, inv), self.weight)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.bias = Parameter(np.zeros(dim, dtype=np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = ops.mean(x, axis=-1, keepdims=True)
        xc = ops.sub(x, mu)
        var = ops.mean(ops.mul(xc, xc), axis=-1, keepdims=True)
        inv = ops.pow(ops.add(var, Tensor(np.float32(self.eps))), -0.5)
        return ops.add(ops.mul(ops.mul(xc, inv), self.weight), self.bias)


class Dropout(Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = (get_rng().uniform(size=x.shape) < keep).astype(np.float32) / keep
        return ops.mul(x, Tensor(mask))


# ─── RoPE ───────────────────────────────────────────────────────────────────


def _rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
    t = np.arange(seq_len, dtype=np.float32)
    ang = np.outer(t, freqs)  # [T, half]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """x: [B, H, T, D]. Rotates pairs (x[..., :D/2], x[..., D/2:])."""
    from axis import backend as _B
    d_half = x.shape[-1] // 2
    x1 = ops.getitem(x, (Ellipsis, slice(0, d_half)))
    x2 = ops.getitem(x, (Ellipsis, slice(d_half, None)))
    c = Tensor(_B.like(x.data, cos[None, None, : x.shape[2], :]))
    s = Tensor(_B.like(x.data, sin[None, None, : x.shape[2], :]))
    r1 = ops.sub(ops.mul(x1, c), ops.mul(x2, s))
    r2 = ops.add(ops.mul(x1, s), ops.mul(x2, c))
    return ops.cat([r1, r2], axis=-1)


# ─── Attention (causal, GQA) ────────────────────────────────────────────────


class CausalSelfAttention(Module):
    """Multi-head causal self-attention with grouped-query attention (GQA)."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 max_seq_len: int = 2048, rope_theta: float = 10000.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.head_dim = dim // n_heads
        self.q_proj = Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(n_heads * self.head_dim, dim, bias=False)
        self._cos, self._sin = _rope_cache(max_seq_len, self.head_dim, rope_theta)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        rep = self.n_heads // self.n_kv_heads

        def split(t: Tensor, heads: int) -> Tensor:
            # [B, T, H*D] -> [B, H, T, D]
            return ops.transpose(ops.reshape(t, (B, T, heads, self.head_dim)), 1, 2)

        q = split(self.q_proj(x), self.n_heads)
        k = split(self.k_proj(x), self.n_kv_heads)
        v = split(self.v_proj(x), self.n_kv_heads)

        q = apply_rope(q, self._cos, self._sin)
        k = apply_rope(k, self._cos, self._sin)

        if rep > 1:
            # Repeat KV heads to match Q heads (GQA expansion).
            k = ops.cat([k] * rep, axis=1)
            v = ops.cat([v] * rep, axis=1)

        # Fused: scores + causal mask + stable softmax + weighted sum in one
        # op (single GPU kernel when acceleration is enabled).
        out = ops.fused_causal_attention(q, k, v, 1.0 / math.sqrt(self.head_dim))
        out = ops.reshape(ops.transpose(out, 1, 2), (B, T, self.n_heads * self.head_dim))
        return self.o_proj(out)


class SwiGLU(Module):
    """Llama-family MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = Linear(dim, hidden, bias=False)
        self.up_proj = Linear(dim, hidden, bias=False)
        self.down_proj = Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Fused silu(gate) * up — one tape node, one GPU round trip.
        return self.down_proj(ops.silu_mul(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, mlp_hidden: int,
                 max_seq_len: int = 2048, rope_theta: float = 10000.0,
                 norm_eps: float = 1e-5):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads, max_seq_len, rope_theta)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def forward(self, x: Tensor) -> Tensor:
        x = ops.add(x, self.attn(self.attn_norm(x)))
        x = ops.add(x, self.mlp(self.mlp_norm(x)))
        return x


class Transformer(Module):
    """Decoder-only transformer (Llama/AQ-5B-class): RoPE+RMSNorm+SwiGLU+GQA."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        mlp_hidden: Optional[int] = None,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.embed = Embedding(vocab_size, dim)
        self.blocks = ModuleList([
            TransformerBlock(
                dim, n_heads, n_kv_heads or n_heads,
                mlp_hidden or int(dim * 8 / 3),
                max_seq_len, rope_theta, norm_eps,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.tie_embeddings = tie_embeddings
        if not tie_embeddings:
            self.lm_head = Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: [B, T] int64 → logits [B, T, V]."""
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.tie_embeddings:
            # Tied LM head: logits = x @ E^T (differentiable through E).
            return ops.matmul(x, ops.transpose(self.embed.weight, 0, 1))
        return self.lm_head(x)

    def loss(self, tokens: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
        logits = self.forward(tokens)
        B, T, V = logits.shape
        return ops.cross_entropy(
            ops.reshape(logits, (B * T, V)),
            ops.reshape(targets, (B * T,)),
            ignore_index=ignore_index,
        )


def _as_weight_t(w: Tensor) -> Tensor:
    """Transposed view of the embedding matrix for tied LM head (differentiable)."""
    return ops.transpose(w, 0, 1)
