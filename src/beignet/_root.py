from typing import Callable, Literal

from torch import Tensor

import beignet


def root(
    func: Callable,
    x0: Tensor,
    x1: Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    method: Literal["chandrupatla", "newton"] = "chandrupatla",
    **kwargs,
):
    r"""
    Find a root of a scalar function.

    Parameters
    ----------
    func : callable
        The function whose root is to be found.

    x0 : Tensor
        The initial guess for the root.

    x1 : Tensor
        The second initial guess for the root.

    rtol : float, optional
        The relative tolerance for the root.

    atol : float, optional
        The absolute tolerance for the root.

    maxiter : int, optional
        The maximum number of iterations.

    method : {"chandrupatla", "newton"}, optional
        The method to use.

    kwargs
        Additional keyword arguments to pass to the solver.

    Returns
    -------
    output : Tensor
        The root of the function.
    """
    match method:
        case "chandrupatla":
            return beignet.chandrupatla(
                func,
                x0,
                x1,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                **kwargs,
            )
        case "newton":
            return beignet.newton(
                func,
                x0,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                **kwargs,
            )
        case _:
            raise ValueError
