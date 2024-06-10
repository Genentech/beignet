from typing import Callable

import torch
from torch import Tensor


def newton(
    func: Callable,
    a: Tensor | None = None,
    *,
    atol: float = 0.000001,
    rtol: float = 0.000001,
    maxiter: int = 50,
) -> (Tensor, (bool, int)):
    r"""
    Find the root of a function using Newton’s method.

    Parameters
    ----------
    func : Callable
        The function for which to find the root.

    a : Tensor, optional
        Initial guess. If not provided, a zero tensor is used.

    atol : float, optional
        Absolute tolerance. Default is 1e-6.

    rtol : float, optional
        Relative tolerance. Default is 1e-6.

    maxiter : int, optional
        Maximum number of iterations. Default is 50.

    Returns
    -------
    output : Tensor
        Root of the function.
    """
    if a is None:
        a = torch.zeros([0])

    for iteration in range(maxiter):
        b = a - torch.linalg.solve(torch.func.jacfwd(func)(a), func(a))

        if torch.linalg.norm(b - a) < atol + rtol * torch.linalg.norm(b):
            return b, (True, iteration)

        a = b

    return b, (False, maxiter)
