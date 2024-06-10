from typing import Callable, Literal, Optional

from torch import Tensor

import beignet


def root(
    func: Callable,
    x0: Tensor,
    x1: Tensor,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    maxiter: int = 100,
    method: Literal["chandrupatla", "newton"] = "chandrupatla",
    **kwargs,
):
    r""" """
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
            )
        case _:
            raise ValueError
