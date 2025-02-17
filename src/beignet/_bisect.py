from typing import Callable

import torch
from torch import Tensor
from torch._higher_order_ops import while_loop

from ._root_scalar import RootSolutionInfo


def bisect(
    func: Callable,
    *args,
    a: float | Tensor,
    b: float | Tensor,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    return_solution_info: bool = False,
    **_,
) -> Tensor | tuple[Tensor, RootSolutionInfo]:
    """Find the root of a scalar (elementwise) function using bisection.

    This method is slow but guarenteed to converge.

    Parameters
    ----------
    func: Callable
        Function to find a root of. Called as `f(x, *args)`.
        The function must operate element wise, i.e. `f(x[i]) == f(x)[i]`.
        Handling *args via broadcasting is acceptable.

    *args
        Extra arguments to be passed to `func`.

    a: float | Tensor
        Lower bracket for root

    b: float | Tensor
        Upper bracket for root

    rtol: float | None = None
        Relative tolerance

    atol: float | None = None
        Absolute tolerance

    maxiter: int = 100
        Maximum number of iterations

    return_solution_info: bool = False
        Whether to return a `RootSolutionInfo` object

    Returns
    -------
    Tensor | tuple[Tensor, RootSolutionInfo]
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    a, b, *args = torch.broadcast_tensors(a, b, *args)

    dtype = a.dtype
    for x in (b, *args):
        dtype = torch.promote_types(x.dtype, dtype)

    eps = torch.finfo(dtype).eps
    a, b, *args = (x.to(dtype=dtype).contiguous() for x in (a, b, *args))

    fa = func(a, *args)
    fb = func(b, *args)

    c = (a + b) / 2
    fc = func(c, *args)

    if rtol is None:
        rtol = eps

    if atol is None:
        atol = 2 * eps

    converged = torch.zeros_like(fa, dtype=torch.bool)
    iterations = torch.zeros_like(fa, dtype=torch.int)

    if (torch.sign(fa) * torch.sign(fb) > 0).any():
        raise ValueError("a and b must bracket a root")

    def condition(a, b, c, fa, fb, fc, converged, iterations):
        return ~converged.all() & (iterations <= maxiter).all()

    def loop_body(a, b, c, fa, fb, fc, converged, iterations):
        cond = torch.sign(fc) == torch.sign(fa)
        a = torch.where(cond, c, a)
        b = torch.where(cond, b, c)
        c = (a + b) / 2
        fc = func(c, *args)
        converged = converged | ((b - a).abs() / 2 < (rtol * torch.abs(c) + atol))
        iterations = iterations + ~converged
        return a, b, c, fa, fb, fc, converged, iterations

    a, b, c, fa, fb, fc, converged, iterations = while_loop(
        condition, loop_body, (a, b, c, fa, fb, fc, converged, iterations)
    )

    if return_solution_info:
        return c, RootSolutionInfo(converged=converged, iterations=iterations)
    else:
        return c
