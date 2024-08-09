from typing import Callable

import torch
from torch import Tensor

from beignet._root_scalar import RootSolutionInfo


def bisect(
    f: Callable,
    *args,
    lower: float,
    upper: float,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    return_solution_info: bool = False,
    dtype=None,
    device=None,
    **_,
) -> Tensor | tuple[Tensor, RootSolutionInfo]:
    a = torch.tensor(lower, dtype=dtype, device=device)
    b = torch.tensor(upper, dtype=dtype, device=device)

    fa = f(a, *args)
    fb = f(b, *args)
    c = (a + b) / 2
    fc = f(c, *args)

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
        fc = f(c, *args)
        iterations += ~converged

    if return_solution_info:
        return c, RootSolutionInfo(converged=converged, iterations=iterations)
    else:
        return c
