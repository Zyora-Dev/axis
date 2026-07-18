"""axis.tokenizer — load a real HuggingFace tokenizer (byte-level BPE).

Reads a model's `tokenizer.json` and reproduces its exact encoding, so a
pretrained model loaded with `axis.from_pretrained` can be fine-tuned on real
text with the right token ids. Covers the byte-level BPE family (GPT-2, Llama-3,
Qwen2, Mistral, ...). Needs the small `regex` package for the \\p{L} pre-token
split (same as HuggingFace's own reference GPT-2 code).

The dependency-free `ByteTokenizer` in axis.data stays available for
from-scratch training and tests.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Sequence

try:
    import regex as _regex
except ImportError:  # pragma: no cover
    _regex = None

# GPT-2 / HF ByteLevel pre-tokenizer split pattern.
_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache()
def _bytes_to_unicode() -> Dict[int, str]:
    """Reversible map from every byte 0-255 to a printable unicode char
    (the classic GPT-2 table)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _get_pairs(word):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class HFTokenizer:
    """Byte-level BPE tokenizer loaded from a HuggingFace tokenizer.json."""

    def __init__(self, vocab: Dict[str, int], merges: Sequence,
                 added_tokens: Sequence[dict] = ()):
        if _regex is None:
            raise ImportError("HFTokenizer needs the `regex` package: pip install regex")
        self.encoder: Dict[str, int] = dict(vocab)
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        # merges may be "a b" strings or [a, b] pairs
        pairs = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges]
        self.bpe_ranks = {p: i for i, p in enumerate(pairs)}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = _regex.compile(_PAT)
        self._cache: Dict[str, str] = {}
        # special / added tokens: match their literal content, emit their id
        self.special_to_id: Dict[str, int] = {}
        self.special_ids: set = set()
        for t in added_tokens:
            self.special_to_id[t["content"]] = t["id"]
            self.special_ids.add(t["id"])
            self.decoder.setdefault(t["id"], t["content"])
        self._special_pat = None
        if self.special_to_id:
            alt = "|".join(_regex.escape(s) for s in sorted(self.special_to_id, key=len, reverse=True))
            self._special_pat = _regex.compile("(" + alt + ")")

    # ── construction ──
    @classmethod
    def from_file(cls, path: str) -> "HFTokenizer":
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        model = spec["model"]
        if "vocab" not in model or "merges" not in model:
            raise ValueError(f"{path}: not a byte-level BPE tokenizer (no vocab/merges)")
        return cls(model["vocab"], model["merges"], spec.get("added_tokens", ()))

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "HFTokenizer":
        return cls.from_file(os.path.join(model_dir, "tokenizer.json"))

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # ── BPE core ──
    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if word[j] == first and j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self._cache[token] = out
        return out

    # ── encode / decode ──
    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in self.pat.findall(text):
            s = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            for piece in self._bpe(s).split(" "):
                ids.append(self.encoder[piece])
        return ids

    def encode(self, text: str) -> List[int]:
        """Text -> token ids (special tokens in the text are honoured)."""
        if not self._special_pat:
            return self._encode_ordinary(text)
        ids: List[int] = []
        for chunk in self._special_pat.split(text):
            if not chunk:
                continue
            if chunk in self.special_to_id:
                ids.append(self.special_to_id[chunk])
            else:
                ids.extend(self._encode_ordinary(chunk))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Token ids -> text."""
        out, buf = [], []
        for i in ids:
            i = int(i)
            if i in self.special_ids:
                if buf:
                    out.append(self._flush(buf)); buf = []
                out.append(self.decoder.get(i, ""))
            else:
                buf.append(i)
        if buf:
            out.append(self._flush(buf))
        return "".join(out)

    def _flush(self, ids: List[int]) -> str:
        text = "".join(self.decoder.get(i, "") for i in ids)
        return bytearray(self.byte_decoder[c] for c in text if c in self.byte_decoder).decode(
            "utf-8", errors="replace")
