"""axis.data — tokenization + data pipeline for language-model training.

Framework-appropriate scope: Axis owns the *training* pipeline (packing,
shuffling, batching, causal-LM target shifting). Tokenizers are pluggable — a
dependency-free `ByteTokenizer` ships for from-scratch training and tests; to
fine-tune a specific pretrained model, feed token ids from that model's own
tokenizer (a bytes/int array is all the pipeline needs).
"""
from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Union

import numpy as np

from axis.tensor import Tensor


class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer (vocab = 256). No dependencies, no
    training — every string round-trips exactly. Good for from-scratch runs
    and deterministic tests."""

    vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


class LMDataset:
    """Packs a token stream into fixed-length causal-LM windows.

    A flat stream of ids is chunked into windows of `seq_len + 1`; each window
    yields (input = window[:-1], target = window[1:]). By default windows are
    contiguous (packing, no wasted tokens); set `stride` for overlap.
    """

    def __init__(self, tokens: Union[Sequence[int], np.ndarray, Iterable],
                 seq_len: int, stride: int | None = None):
        if not isinstance(tokens, np.ndarray):
            tokens = np.fromiter(tokens, dtype=np.int64)
        self.tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride else self.seq_len
        if self.tokens.size < seq_len + 1:
            raise ValueError(
                f"need > seq_len+1={seq_len+1} tokens, got {self.tokens.size}")
        # start index of each window
        last = self.tokens.size - (seq_len + 1)
        self._starts = np.arange(0, last + 1, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._starts.size)

    def __getitem__(self, i: int):
        s = int(self._starts[i])
        w = self.tokens[s : s + self.seq_len + 1]
        return w[:-1], w[1:]


class DataLoader:
    """Batches an LMDataset into (input, target) Tensors of shape [B, seq_len].

    Deterministic shuffling via a seeded RNG (same seed -> same order), so runs
    are reproducible. `drop_last` keeps every batch full (default True).
    """

    def __init__(self, dataset: LMDataset, batch_size: int, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True):
        self.ds = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.ds)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        order = np.arange(len(self.ds))
        if self.shuffle:
            self._rng.shuffle(order)
        bs = self.batch_size
        n = len(self.ds)
        end = (n // bs) * bs if self.drop_last else n
        for start in range(0, end, bs):
            idx = order[start : start + bs]
            xs, ys = zip(*(self.ds[int(i)] for i in idx))
            yield (Tensor(np.stack(xs).astype(np.int64)),
                   Tensor(np.stack(ys).astype(np.int64)))
