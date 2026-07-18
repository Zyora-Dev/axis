"""axis.generate — autoregressive text generation from a trained model.

Greedy / temperature / top-k sampling, deterministic via a seeded RNG. Runs
under no_grad (inference only). No KV cache yet — the full context is re-run
each step (O(T^2)); correct and simple, which is the priority.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from axis.tensor import Tensor, no_grad


def _model_context(model) -> int:
    """The model's max sequence length, read from the RoPE cache."""
    try:
        return int(model.blocks[0].attn._cos.shape[0])
    except Exception:  # noqa: BLE001
        return 512


def generate(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: int = 0,
    eos_id: Optional[int] = None,
) -> List[int]:
    """Continue `prompt_ids` for up to `max_new_tokens`.

    temperature=0 → greedy argmax. Otherwise sample from softmax(logits/T),
    optionally restricted to the top_k logits. Stops early on `eos_id`.
    Returns the full sequence (prompt + generated).
    """
    model.eval()
    ctx_len = _model_context(model)
    rng = np.random.default_rng(seed)
    ids: List[int] = [int(t) for t in prompt_ids]

    with no_grad():
        for _ in range(max_new_tokens):
            ctx = ids[-ctx_len:]
            logits = model.forward(Tensor(np.array([ctx], dtype=np.int64))).data[0, -1]
            logits = logits.astype(np.float32)

            if temperature <= 0.0:
                nxt = int(np.argmax(logits))
            else:
                logits = logits / float(temperature)
                if top_k is not None and 0 < top_k < logits.size:
                    kth = np.partition(logits, -top_k)[-top_k]
                    logits = np.where(logits < kth, -np.inf, logits)
                logits -= logits.max()
                p = np.exp(logits)
                p /= p.sum()
                nxt = int(rng.choice(logits.size, p=p))

            ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
    return ids
