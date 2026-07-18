"""Axis Tensor — the core autograd tensor.

Design principles (reliability first):
- Tape-based reverse-mode autodiff. Every op records a node with an exact
  backward closure. `backward()` walks the tape in reverse topological order.
- float32 by default (training dtype in Phase 1); int64 for indices.
- Deterministic: axis.manual_seed() seeds a private Generator used by all
  random initializers — two runs with the same seed are bit-identical.
- No silent broadcasting bugs: gradients are always reduced back to the
  parameter's shape via `_unbroadcast`.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

# ─── Global RNG (determinism) ───────────────────────────────────────────────

_rng = np.random.default_rng(0)


def manual_seed(seed: int) -> None:
    """Seed Axis's private RNG. Same seed → bit-identical runs."""
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _rng


# ─── grad-mode (no_grad) ────────────────────────────────────────────────────

_grad_enabled = threading.local()


def is_grad_enabled() -> bool:
    return getattr(_grad_enabled, "value", True)


@contextlib.contextmanager
def no_grad():
    """Disable tape recording (inference / optimizer updates)."""
    prev = is_grad_enabled()
    _grad_enabled.value = False
    try:
        yield
    finally:
        _grad_enabled.value = prev


# ─── Tensor ─────────────────────────────────────────────────────────────────


class Tensor:
    """An n-dimensional array with reverse-mode autodiff.

    Attributes:
        data: the underlying numpy array (float32 or int64).
        grad: accumulated gradient (numpy array, same shape) or None.
        requires_grad: whether this tensor participates in autograd.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_parents", "_op", "_dev")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _parents: Sequence["Tensor"] = (),
        _op: str = "",
    ):
        if isinstance(data, Tensor):
            data = data.data
        from axis import backend as _B
        if _B.is_gpu_array(data):
            arr = data                       # keep on GPU
        else:
            arr = np.asarray(data)
        if arr.dtype in (np.float64, np.float16):
            arr = arr.astype(np.float32)
        elif arr.dtype in (np.int32, np.int8, np.uint8, np.int16):
            arr = arr.astype(np.int64)
        elif arr.dtype == np.bool_:
            arr = arr.astype(np.float32)
        # Move onto the active device so constants/batches don't stay on host
        # while the model is on the GPU (cupy rejects mixed numpy+cupy ops).
        if _B.device() == "gpu" and not _B.is_gpu_array(arr):
            arr = _B.to_gpu_array(arr)
        self.data = arr
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = bool(requires_grad) and is_grad_enabled()
        self._backward: Optional[Callable[[np.ndarray], None]] = None
        self._parents: tuple[Tensor, ...] = tuple(_parents)
        self._op: str = _op
        # Optional cached GPU buffer (device residency). `.data` is always the
        # source of truth; `_dev` just avoids re-uploading a value that is
        # already on the GPU (e.g. a shared activation used by several matmuls).
        self._dev = None

    # ── properties ──
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        from axis import ops
        return ops.transpose(self)

    def numpy(self) -> np.ndarray:
        from axis import backend as _B
        return _B.to_numpy(self.data)

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def __repr__(self) -> str:
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype={self.data.dtype}{grad_str}, op={self._op!r})"

    def __len__(self) -> int:
        return self.shape[0]

    # ── autograd core ──
    def backward(self, grad: Optional[np.ndarray] = None,
                 retain_graph: bool = False) -> None:
        """Reverse-mode autodiff from this tensor through the tape.

        With `retain_graph=False` (the default, matching PyTorch) each node's
        saved activation and backward closure are released as soon as its
        gradient has been propagated — this bounds peak memory during backward
        instead of holding the whole graph alive. Pass `retain_graph=True` to
        keep the graph for a second backward pass.
        """
        if not self.requires_grad and self._backward is None:
            raise RuntimeError("backward() on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("backward() without grad only allowed on scalars")
            from axis import backend as _B
            grad = _B.array_module(self.data).ones_like(self.data)
        else:
            from axis import backend as _B
            grad = _B.array_module(self.data).asarray(grad, dtype=self.data.dtype)

        # Topological order (iterative — no recursion limit issues on deep nets).
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))

        grads: dict[int, np.ndarray] = {id(self): grad}
        for node in reversed(topo):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node.requires_grad and not node._parents:
                # Leaf: accumulate into .grad
                if node.grad is None:
                    node.grad = g.copy()
                else:
                    node.grad += g
            if node._backward is not None:
                for parent, pg in node._backward(g):
                    if pg is None:
                        continue
                    key = id(parent)
                    if key in grads:
                        grads[key] = grads[key] + pg
                    else:
                        grads[key] = pg
                # Free this node's saved graph state now that its backward has
                # run: the activation is no longer needed (all downstream
                # consumers were processed earlier in reverse topo, and this
                # node's own backward just used it). Leaves (params/inputs) and
                # the root tensor are preserved.
                if not retain_graph:
                    node._backward = None
                    if node._parents and node is not self:
                        node.data = None      # release the activation array
                        node._dev = None      # and any cached device buffer
                    node._parents = ()

    def zero_grad(self) -> None:
        self.grad = None

    # ── operator sugar (delegates to ops) ──
    def __add__(self, other):
        from axis import ops
        return ops.add(self, _wrap(other))

    __radd__ = __add__

    def __mul__(self, other):
        from axis import ops
        return ops.mul(self, _wrap(other))

    __rmul__ = __mul__

    def __sub__(self, other):
        from axis import ops
        return ops.sub(self, _wrap(other))

    def __rsub__(self, other):
        from axis import ops
        return ops.sub(_wrap(other), self)

    def __truediv__(self, other):
        from axis import ops
        return ops.div(self, _wrap(other))

    def __rtruediv__(self, other):
        from axis import ops
        return ops.div(_wrap(other), self)

    def __neg__(self):
        from axis import ops
        return ops.mul(self, Tensor(np.float32(-1.0)))

    def __pow__(self, p):
        from axis import ops
        return ops.pow(self, float(p))

    def __matmul__(self, other):
        from axis import ops
        return ops.matmul(self, _wrap(other))

    def __getitem__(self, idx):
        from axis import ops
        return ops.getitem(self, idx)

    # ── shape sugar ──
    def reshape(self, *shape):
        from axis import ops
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return ops.reshape(self, shape)

    def transpose(self, a: int = -2, b: int = -1):
        from axis import ops
        return ops.transpose(self, a, b)

    def sum(self, axis=None, keepdims: bool = False):
        from axis import ops
        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False):
        from axis import ops
        return ops.mean(self, axis=axis, keepdims=keepdims)


def _wrap(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=np.float32))


# ─── constructors ───────────────────────────────────────────────────────────


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False, std: float = 1.0) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(
        (_rng.standard_normal(shape) * std).astype(np.float32),
        requires_grad=requires_grad,
    )


def arange(*args, requires_grad: bool = False) -> Tensor:
    return Tensor(np.arange(*args, dtype=np.int64), requires_grad=requires_grad)
