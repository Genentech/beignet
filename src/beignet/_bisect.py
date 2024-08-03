from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class RootSolutionInfo:
    converged: Tensor
    iterations: Tensor


def implicit_differentiation_wrapper(solver: Callable[..., Tensor]):
    def inner(f, *args, **kwargs):
        class SolverWrapper(torch.autograd.Function):
            @staticmethod
            def forward(*args):
                def g(x):
                    return f(x, *args)

                return solver(g, **kwargs)

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(output, *inputs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                xstar, *args = ctx.saved_tensors
                n_args = len(args)

                # f(x^*(theta), theta) = 0

                A, *B = torch.func.jacrev(f, argnums=tuple(range(n_args + 1)))(
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
                def g(x: Tensor) -> Tensor:
                    x, *args_ = torch.broadcast_tensors(x, *args)
                    return torch.func.vmap(
                        lambda x, *args: f(x, *args),
                        in_dims=(0, *in_dims),
                    )(x, *args_)

                out = solver(g, **kwargs)
                return out, 0

        return SolverWrapper.apply(*args)

    return inner


@implicit_differentiation_wrapper
def bisect(
    f: Callable,
    *,
    lower: float,
    upper: float,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    dtype=None,
    device=None,
    **_,
) -> Tensor:
    a = torch.tensor(lower, dtype=dtype, device=device)
    b = torch.tensor(upper, dtype=dtype, device=device)

    fa = f(a)
    fb = f(b)
    c = (a + b) / 2
    fc = f(c)

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
        fc = f(c)
        iterations += ~converged

    return c
