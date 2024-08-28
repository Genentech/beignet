from typing import Callable

import torch
from torch import Tensor

from ._root_scalar import RootSolutionInfo


def bisect(
    func: Callable,
    *args,
    lower: float | Tensor,
    upper: float | Tensor,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    return_solution_info: bool = False,
    dtype=None,
    device=None,
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

    lower: float | Tensor
        Lower bracket for root

    upper: float | Tensor
        Upper bracket for root

    rtol: float | None = None
        Relative tolerance

    atol: float | None = None
        Absolute tolerance

    maxiter: int = 100
        Maximum number of iterations

    return_solution_info: bool = False
        Whether to return a `RootSolutionInfo` object

    dtype = None
        if upper/lower are passed as floats instead of tensors
        use this dtype when constructing the tensor.

    device = None
        if upper/lower are passed as floats instead of tensors
        use this device when constructing the tensor.

    Returns
    -------
    Tensor | tuple[Tensor, RootSolutionInfo]
    """
    a = torch.as_tensor(lower, dtype=dtype, device=device)
    b = torch.as_tensor(upper, dtype=dtype, device=device)
    a, b, *args = torch.broadcast_tensors(a, b, *args)

    fa = func(a, *args)
    fb = func(b, *args)

    c = (a + b) / 2
    fc = func(c, *args)

    eps = torch.finfo(fa.dtype).eps

    if rtol is None:
        rtol = eps

    if atol is None:
        atol = 2 * eps

    converged = torch.zeros_like(fa, dtype=torch.bool)
    iterations = torch.zeros_like(fa, dtype=torch.int)

    if (torch.sign(fa) * torch.sign(fb) > 0).any():
        raise ValueError("a and b must bracket a root")

    for _ in range(maxiter):
        converged = converged | ((b - a) / 2 < (rtol * torch.abs(c) + atol))

        if converged.all():
            break

        cond = torch.sign(fc) == torch.sign(fa)
        a = torch.where(cond, c, a)
        b = torch.where(cond, b, c)
        c = (a + b) / 2
        fc = func(c, *args)
        iterations = iterations + ~converged

    if return_solution_info:
        return c, RootSolutionInfo(converged=converged, iterations=iterations)
    else:
        return c
