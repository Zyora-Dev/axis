"""Tests for LoRA fine-tuning: zero-init identity, only adapters train, base
weights stay frozen, training lowers loss, and merge folds correctly.
"""
import numpy as np

import axis
from axis import nn, optim, lora
from axis.tensor import Tensor


def _model():
    axis.manual_seed(0)
    return nn.Transformer(vocab_size=128, dim=64, n_layers=2, n_heads=4,
                          n_kv_heads=2, mlp_hidden=128, max_seq_len=64,
                          tie_embeddings=True)


def test_lora_zero_init_is_identity():
    m = _model()
    toks = Tensor(np.array([[3, 7, 1, 9]], dtype=np.int64))
    before = m.forward(toks).data.copy()
    lora.apply_lora(m, rank=8, alpha=16)
    after = m.forward(toks).data
    # B initialised to zero -> adapted model == base model at step 0
    assert np.allclose(before, after, rtol=1e-5, atol=1e-5)


def test_lora_only_adapters_trainable():
    m = _model()
    lora.apply_lora(m, rank=4, alpha=8)
    trainable = lora.trainable_parameters(m)
    assert len(trainable) > 0
    # every trainable param is a LoRA adapter
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all("lora_a" in n or "lora_b" in n for n in names)
    # adapters are a small fraction of total params
    tot = m.num_parameters()
    adapt = sum(p.size for p in trainable)
    assert adapt < 0.25 * tot


def test_lora_trains_and_base_frozen():
    m = _model()
    # snapshot a base weight before training
    base_w = dict(m.named_parameters())["blocks.0.attn.q_proj.weight"].data.copy()
    lora.apply_lora(m, rank=8, alpha=16)
    opt = optim.AdamW(lora.trainable_parameters(m), lr=1e-3)

    rng = np.random.default_rng(0)
    toks = rng.integers(0, 128, size=(4, 16)).astype(np.int64)
    inp, tgt = Tensor(toks[:, :-1]), Tensor(toks[:, 1:])
    first = last = None
    for _ in range(15):
        loss = m.loss(inp, tgt)
        loss.backward()
        opt.step(); opt.zero_grad()
        first = first if first is not None else float(loss.data)
        last = float(loss.data)
    assert last < first, f"LoRA didn't train: {first} -> {last}"

    # base weight must be unchanged (frozen)
    now_w = dict(m.named_parameters())["blocks.0.attn.q_proj.base.weight"].data
    assert np.array_equal(base_w, now_w), "base weight changed — not frozen!"


def test_merge_lora_matches():
    m = _model()
    lora.apply_lora(m, rank=8, alpha=16)
    # perturb an adapter so the update is non-zero
    for n, p in m.named_parameters():
        if "lora_b" in n:
            p.data = (np.random.default_rng(1).standard_normal(p.data.shape) * 0.02).astype(np.float32)
    toks = Tensor(np.array([[2, 4, 6, 8, 10]], dtype=np.int64))
    pre = m.forward(toks).data.copy()
    lora.merge_lora(m)
    post = m.forward(toks).data
    assert np.allclose(pre, post, rtol=1e-4, atol=1e-4), \
        f"merge changed output: max|Δ|={np.abs(pre - post).max()}"


def test_lora_state_dict_only_adapters():
    m = _model()
    lora.apply_lora(m, rank=4, alpha=8)
    sd = lora.lora_state_dict(m)
    assert sd and all("lora_a" in k or "lora_b" in k for k in sd)
