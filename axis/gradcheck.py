"""axis.gradcheck — numerical gradient verification.

The reliability backbone: every op's analytic backward is compared against
central finite differences. Any future backend (locomp GPU) must pass the same
checks against this NumPy reference.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from axis.tensor import Tensor


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-3,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    verbose: bool = False,
) -> bool:
    """Check analytic grads of `fn(*inputs)` (must return a scalar Tensor).

    Uses float64 finite differences internally for accuracy while the forward
    runs in float32 — tolerances chosen accordingly.
    """
    # Analytic pass.
    for t in inputs:
        t.grad = None
    out = fn(*inputs)
    if out.size != 1:
        raise ValueError("gradcheck: fn must return a scalar (e.g. .sum())")
    out.backward()
    analytic = [t.grad.copy() if t.grad is not None else np.zeros_like(t.data) for t in inputs]

    # Numerical pass (central differences).
    for t_idx, t in enumerate(inputs):
        if not t.requires_grad:
            continue
        num = np.zeros_like(t.data, dtype=np.float64)
        flat = t.data.reshape(-1)
        num_flat = num.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            plus = float(fn(*inputs).data)
            flat[i] = orig - eps
            minus = float(fn(*inputs).data)
            flat[i] = orig
            num_flat[i] = (plus - minus) / (2.0 * eps)

        ok = np.allclose(analytic[t_idx], num, rtol=rtol, atol=atol)
        if not ok:
            diff = np.abs(analytic[t_idx] - num)
            worst = np.unravel_index(diff.argmax(), diff.shape)
            msg = (
                f"gradcheck FAILED for input {t_idx}: "
                f"max diff {diff.max():.6f} at {worst} "
                f"(analytic={analytic[t_idx][worst]:.6f}, numeric={num[worst]:.6f})"
            )
            if verbose:
                print(msg)
            raise AssertionError(msg)
        if verbose:
            print(f"gradcheck OK for input {t_idx} (max |a-n| = {np.abs(analytic[t_idx] - num).max():.2e})")
    return True
