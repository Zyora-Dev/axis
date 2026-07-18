"""Tests for the data pipeline: tokenizer round-trip, causal-LM target shift,
deterministic batching, and an end-to-end training step that lowers the loss.
"""
import numpy as np

import axis
from axis import nn, optim
from axis.data import ByteTokenizer, LMDataset, DataLoader


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    for s in ["hello world", "Axis 🚀 ünïcode", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_lmdataset_shift():
    ds = LMDataset(list(range(10)), seq_len=4)
    x, y = ds[0]
    assert list(x) == [0, 1, 2, 3]
    assert list(y) == [1, 2, 3, 4]      # target = input shifted by one
    # contiguous packing (stride = seq_len)
    x1, _ = ds[1]
    assert list(x1) == [4, 5, 6, 7]


def test_dataloader_shapes_and_determinism():
    ds = LMDataset(np.arange(200), seq_len=8)
    dl1 = DataLoader(ds, batch_size=4, shuffle=True, seed=42)
    dl2 = DataLoader(ds, batch_size=4, shuffle=True, seed=42)
    b1 = [(x.data.copy(), y.data.copy()) for x, y in dl1]
    b2 = [(x.data.copy(), y.data.copy()) for x, y in dl2]
    assert b1[0][0].shape == (4, 8)
    # same seed -> identical order
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(b1, b2))


def test_training_step_lowers_loss():
    axis.manual_seed(0)
    text = "the quick brown fox jumps over the lazy dog. " * 40
    ids = ByteTokenizer().encode(text)
    ds = LMDataset(ids, seq_len=16)
    dl = DataLoader(ds, batch_size=4, shuffle=True, seed=0)

    model = nn.Transformer(vocab_size=256, dim=64, n_layers=2, n_heads=4,
                           n_kv_heads=2, mlp_hidden=128, max_seq_len=64)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    first = None
    last = None
    for epoch in range(3):
        for inp, tgt in dl:
            loss = model.loss(inp, tgt)
            loss.backward()
            opt.step(); opt.zero_grad()
            if first is None:
                first = float(loss.data)
            last = float(loss.data)
    assert last < first, f"loss did not drop: {first} -> {last}"
