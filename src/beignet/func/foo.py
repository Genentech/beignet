from typing import Any, Callable, Tuple

import torch
from torch import Tensor
from torch.autograd.function import Function


def root(
    func: Callable[[Tensor], Tensor],
    x0: Tensor,
    f: Callable[[Callable, Tensor], Tensor],
    g: Callable[[Callable, Tensor], Tensor],
) -> Tensor:
    class Root(Function):
        @staticmethod
        def forward(*args: Any, **kwargs: Any) -> Any:
            return f(func, *args)

        @staticmethod
        def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
            (output,) = output

            ctx.save_for_backward(output)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            (output,) = ctx.saved_tensors

            _, jvp_fn = torch.func.linearize(func, output)

            def fn(x):
                return jvp_fn(x)

            return g(fn, *grad_outputs)

        @staticmethod
        def jvp(ctx: Any, *grad_inputs: Any) -> Any:
            pass

        # @staticmethod
        # def jvp(ctx, *tangents):
        #     x0 = ctx.x0
        #
        #     tangent, = tangents
        #
        #     _, jvp_fn = torch.func.linearize(func, x0)
        #
        #     return jvp_fn(tangent)

    return Root.apply(x0)
