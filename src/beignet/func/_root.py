from typing import Any, Callable, Tuple

import torch
from torch import Tensor
from torch.autograd import Function


def root(
    func: Callable[[Tensor], Tensor],
    x0: Tensor,
    f: Callable[[Callable, Tensor], Tensor],
    g: Callable[[Callable, Tensor], Tensor],
) -> Tensor:
    class Root(Function):
        @staticmethod
        def forward(*args: Any, **kwargs: Any) -> Tensor:
            return f(func, *args, **kwargs)

        @staticmethod
        def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any):
            (ouput,) = output

            ctx.save_for_backward(ouput)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            (output,) = ctx.saved_tensors

            _, jvp_fn = torch.func.linearize(func, output)

            def fn(x):
                return jvp_fn(x)

            return g(fn, *grad_outputs)

    return Root.apply(x0)
