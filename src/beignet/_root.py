from typing import Callable, Literal

import torch
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


def implicit_differentiation_wrapper(solver: Callable[..., Tensor]):
    def inner(f, *args, **kwargs):
        class SolverWrapper(torch.autograd.Function):
            @staticmethod
            def forward(*args):
                return solver(f, *args, **kwargs)

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(output, *inputs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                xstar, *args = ctx.saved_tensors
                nargs = len(args)

                # optimality condition:
                # f(x^*(theta), theta) = 0

                A, *B = torch.func.jacrev(f, argnums=tuple(range(nargs + 1)))(
                    xstar, *args
                )

                if A.ndim == 0:
                    return tuple(
                        -g * b / A for g, b in zip(grad_outputs, B, strict=True)
                    )
                elif A.ndim == 2:
                    return tuple(
                        torch.linalg.solve(A, -g * b)
                        for g, b in zip(grad_outputs, B, strict=True)
                    )
                else:
                    raise RuntimeError(f"{A.ndim=} != 0 or 2")

            @staticmethod
            def vmap(info, in_dims, *args):
                def g(x: Tensor, *args) -> Tensor:
                    return torch.func.vmap(
                        lambda x, *args: f(x, *args),
                        in_dims=(0, *in_dims),
                    )(*torch.broadcast_tensors(x, *args))

                out = solver(g, *args, **kwargs)
                return out, 0

        return SolverWrapper.apply(*args)

    return inner


def root_scalar(
    f: Callable,
    *args,
    method: Literal["bisect"] = "bisect",
    implicit_diff: bool = True,
    options: dict,
):
    if method == "bisect":
        solver = beignet.bisect
    else:
        raise ValueError(f"method {method} not recognized")

    if implicit_diff:
        solver = implicit_differentiation_wrapper(solver)

    return solver(f, *args, **options)
