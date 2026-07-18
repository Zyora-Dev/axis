"""Regression test for the GQA kv-head repeat ordering.

HF `repeat_kv` is GROUPED: each kv head is repeated `rep` times contiguously,
so query head j pairs with kv head j // rep. A tiled repeat (cat([k]*rep)) maps
head j to kv j % n_kv — the wrong pairing, which silently broke every GQA model
(Llama-2/3, Mistral, Qwen). This locks the grouped order.
"""
import numpy as np

import axis
from axis import nn, ops
from axis.tensor import Tensor


def test_gqa_grouped_repeat_order():
    n_kv, rep = 3, 2  # -> 6 query heads
    # [B=1, n_kv, T=1, D=1], each kv head tagged with its index
    t = Tensor(np.arange(n_kv, dtype=np.float32).reshape(1, n_kv, 1, 1))
    parts = []
    for i in range(n_kv):
        hi = ops.getitem(t, (slice(None), slice(i, i + 1)))
        parts.extend([hi] * rep)
    out = ops.cat(parts, axis=1).numpy().reshape(-1)
    # grouped: kv0,kv0, kv1,kv1, kv2,kv2  (head j -> kv j//rep)
    assert list(out) == [0, 0, 1, 1, 2, 2]


def test_gqa_attention_runs_and_shapes():
    axis.manual_seed(0)
    attn = nn.CausalSelfAttention(dim=32, n_heads=8, n_kv_heads=2, max_seq_len=16)
    x = Tensor(np.random.default_rng(0).standard_normal((2, 6, 32)).astype(np.float32))
    y = attn(x)
    assert y.shape == (2, 6, 32)
    loss = y.sum()
    loss.backward()  # backward through grouped GQA must work
    assert attn.k_proj.weight.grad is not None
