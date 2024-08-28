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

                xstar, *args = torch.atleast_1d(xstar, *args)
                xstar, *args = torch.broadcast_tensors(xstar, *args)
                shape = xstar.shape

                xstar = xstar.view(-1)
                args = (arg.view(-1) for arg in args)

                argnums = tuple(range(nargs + 1))

                # optimality condition:
                # f(x^*(theta), theta) = 0

                # because f is applied elementwise just compute diagonal of jacobian
                a, *b = torch.vmap(
                    torch.func.grad(f, argnums=argnums), in_dims=(0,) * (nargs + 1)
                )(xstar, *args)

                output = tuple(
                    (-g * b2 / a).view(*shape)
                    for g, b2 in zip(grad_outputs, b, strict=True)
                )

                return output

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
