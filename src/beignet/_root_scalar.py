from typing import Callable, Literal

from torch import Tensor

import beignet
import beignet.func


def root_scalar(
    func: Callable,
    *args,
    method: Literal["bisect", "chandrupatla"] = "chandrupatla",
    implicit_diff: bool = True,
    options: dict | None = None,
) -> Tensor | tuple[Tensor, dict]:
    """
    Find the root of a scalar (elementwise) function.

    Parameters
    ----------
    func: Callable
        Function to find a root of. Called as `f(x, *args)`.
        The function must operate element wise, i.e. `f(x[i]) == f(x)[i]`.
        Handling *args via broadcasting is acceptable.

    *args
        Extra arguments to be passed to `func`.

    method: Literal["bisect", "chandrupatla"] = "chandrupatla"
        Solver method to use.
        * bisect: `beignet.bisect`
        * chandrupatla: `beignet.chandrupatla`
        See docstring of underlying solvers for description of options dict.

    implicit_diff: bool = True
        If true, the solver is wrapped in `beignet.func.custom_scalar_root` which
        enables gradients with respect to *args using implicit differentiation.

    options: dict | None = None
        A dictionary of options that are passed through to the solver as keyword args.


    Returns
    -------
    Tensor | tuple[Tensor, dict]
        Tensor approximately satisfying `f(x^*, *args) = 0`.
        A dictionary of solution metadata may also be returned depending on `options`.
    """
    if options is None:
        options = {}

    if method == "bisect":
        solver = beignet.bisect
    elif method == "chandrupatla":
        solver = beignet.chandrupatla
    else:
        raise ValueError(f"method {method} not recognized")

    if implicit_diff:
        solver = beignet.func.custom_scalar_root(solver)

    return solver(func, *args, **options)
