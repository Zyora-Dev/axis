"""Axis elementwise GPU kernels (locomp).

Only flat, index-by-program_id kernels live here — no constexpr used in
pointer arithmetic, so they coexist safely in one module. Kernels that use
constexpr dims for indexing (matmul, attention, softmax) are isolated in their
own modules to avoid locomp's cross-kernel constexpr type leak.
"""
import locomp


@locomp.kernel
def silu_kernel(X: locomp.Tensor, O: locomp.Tensor):
    i = locomp.program_id(0)
    v = locomp.load(X + i)
    locomp.store(O + i, v / (1.0 + locomp.exp(-v)))


@locomp.kernel
def gelu_kernel(X: locomp.Tensor, O: locomp.Tensor):
    i = locomp.program_id(0)
    x = locomp.load(X + i)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = locomp.tanh(inner)
    locomp.store(O + i, 0.5 * x * (1.0 + t))


@locomp.kernel
def silu_mul_kernel(G: locomp.Tensor, U: locomp.Tensor, O: locomp.Tensor):
    i = locomp.program_id(0)
    g = locomp.load(G + i)
    locomp.store(O + i, (g / (1.0 + locomp.exp(-g))) * locomp.load(U + i))
