"""Tests for the HuggingFace byte-level BPE tokenizer.

The hermetic test builds a tiny tokenizer directly (no network). The GPT-2 test
downloads the real tokenizer.json and checks known encodings; it is skipped if
`regex` is missing or the download fails.
"""
import json
import struct
import urllib.request

import pytest

regex = pytest.importorskip("regex")

from axis.tokenizer import HFTokenizer, _bytes_to_unicode


def test_bpe_merge_and_roundtrip():
    be = _bytes_to_unicode()
    a = be[ord("a")]  # byte-space char for 'a'
    # vocab: single 'a', merged 'aa'; one merge rule (a a) -> aa
    vocab = {a: 0, a + a: 1}
    tok = HFTokenizer(vocab, merges=[f"{a} {a}"])
    # "aaa" -> greedy BPE -> ["aa", "a"] -> [1, 0]
    assert tok.encode("aaa") == [1, 0]
    assert tok.decode([1, 0]) == "aaa"
    assert tok.decode(tok.encode("aa")) == "aa"


def test_special_tokens():
    be = _bytes_to_unicode()
    a, b = be[ord("a")], be[ord("b")]
    vocab = {a: 0, b: 1}
    tok = HFTokenizer(vocab, merges=[],
                      added_tokens=[{"id": 99, "content": "<eos>", "special": True}])
    ids = tok.encode("a<eos>b")
    assert ids == [0, 99, 1]
    assert tok.decode(ids) == "a<eos>b"


@pytest.mark.parametrize("text,expected", [
    ("hello world", [31373, 995]),
    (" hello world", [23748, 995]),
    ("The quick brown fox.", [464, 2068, 7586, 21831, 13]),
])
def test_gpt2_known_encodings(tmp_path, text, expected):
    path = tmp_path / "tok.json"
    try:
        urllib.request.urlretrieve(
            "https://huggingface.co/gpt2/resolve/main/tokenizer.json", path)
    except Exception:  # noqa: BLE001 — offline
        pytest.skip("no network for gpt2 tokenizer")
    tok = HFTokenizer.from_file(str(path))
    assert tok.encode(text) == expected
    assert tok.decode(tok.encode(text)) == text
