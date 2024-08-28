from typing import Callable

import torch
from torch import Tensor
from torch.autograd import Function


def custom_scalar_root(func: Callable[..., Tensor]):
    """Wrap a scalar root solver in an autograd function.

    Parameters
    ----------
    func: Callable[..., Tensor]
        A scalar root solver.
        `solver(f, *args, **kwargs)` should return a root of `f`.
        Gradients can be computed with respect to *args.
    """

    def inner(f, *args, **kwargs):
        class SolverWrapper(Function):
            @staticmethod
            def forward(*args):
                return func(f, *args, **kwargs)

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(output, *inputs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                xstar, *args = ctx.saved_tensors
                nargs = len(args)

                # optimality condition:
                # f(x^*(theta), theta) = 0

                argnums = tuple(range(nargs + 1))

                a, *b = torch.func.jacrev(f, argnums=argnums)(xstar, *args)

                match a.ndim:
                    case 0:
                        output = ()

                        for g, b2 in zip(grad_outputs, b, strict=True):
                            output = (*output, -g * b2 / a)

                        return output
                    case 2:  # NOTE: `a` is diagonal because `f` is scalar
                        output = ()

                        for g, b2 in zip(grad_outputs, b, strict=True):
                            output = (*output, torch.linalg.solve(a, -g * b2))

                        return output
                    case _:
                        raise ValueError

            @staticmethod
            def vmap(info, in_dims, *args):
                # push vmap into function being evaluated
                # inner function needs to also be vmaped over x
                in_dims = (0, *in_dims)
                out = func(
                    torch.vmap(
                        f,
                        in_dims=in_dims,
                        out_dims=0,
                        randomness=info.randomness,
                        chunk_size=info.batch_size,
                    ),
                    *args,
                    **kwargs,
                )

                return out, 0

        return SolverWrapper.apply(*args)

    return inner
