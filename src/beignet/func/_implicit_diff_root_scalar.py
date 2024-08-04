from typing import Callable

import torch
from torch import Tensor


def implicit_diff_root_scalar(solver: Callable[..., Tensor]):
    """Wrap a scalar root solver in an autograd function.

    Parameters
    ----------
    solver: Callable[..., Tensor]
        A scalar root solver.
        `solver(f, *args, **kwargs)` should return a root of `f`.
        Gradients can be computed with respect to *args.
    """

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
                    # NOTE even in this case we should always have A diagonal because f is scalar
                    return tuple(
                        torch.linalg.solve(A, -g * b)
                        for g, b in zip(grad_outputs, B, strict=True)
                    )
                else:
                    raise RuntimeError(f"{A.ndim=} != 0 or 2")

            @staticmethod
            def vmap(info, in_dims, *args):
                # vmap is trivial because in scalar case we just broadcast
                out = solver(f, *args, **kwargs)
                return out, 0

        return SolverWrapper.apply(*args)

    return inner
