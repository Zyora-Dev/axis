"""Isolated row-softmax kernel (uses D constexpr in pointer arithmetic)."""
import locomp


@locomp.kernel
def softmax_rows(X: locomp.Tensor, O: locomp.Tensor, D: locomp.constexpr):
    row = locomp.program_id(0)
    mx = locomp.load(X + row * D)
    for i in range(1, D):
        v = locomp.load(X + row * D + i)
        mx = locomp.max(mx, v)
    s = 0.0
    for i in range(D):
        s = s + locomp.exp(locomp.load(X + row * D + i) - mx)
    for i in range(D):
        e = locomp.exp(locomp.load(X + row * D + i) - mx)
        locomp.store(O + row * D + i, e / s)
