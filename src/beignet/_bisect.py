from typing import Callable

import torch
from torch import Tensor


def bisect(
    f: Callable,
    a: Tensor,
    b: Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    **_,
):
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

    return c, (converged, iterations)


def root_scalar(
    f: Callable,
    *args,
    a: Tensor,
    b: Tensor,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int = 100,
    **_,
):
    class Root(torch.autograd.Function):
        #        generate_vmap_rule = True  # FIXME this doesn't work

        @staticmethod
        def forward(*args):
            root, _ = bisect(
                lambda x: f(x, *args), a, b, rtol=rtol, atol=atol, maxiter=maxiter
            )
            return root

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(output, *inputs)

        @staticmethod
        def backward(ctx, *grad_outputs):
            root, *args = ctx.saved_tensors
            n_args = len(args)

            args = tuple(torch.atleast_1d(x) for x in args)
            root = torch.atleast_1d(root)

            # f(x^*(theta), theta) = 0

            # NOTE jacobian is diagonal b/c f must be a scalar function
            A, *B = torch.func.vmap(
                torch.func.grad(f, argnums=tuple(range(n_args + 1))),
                in_dims=(0,) * (n_args + 1),
            )(root, *args)

            return tuple(-g * b / A for g, b in zip(grad_outputs, B, strict=True))

        @staticmethod
        def vmap(info, in_dims, *args):
            _a, _b, *args = torch.broadcast_tensors(a, b, *args)
            f_ = torch.func.vmap(f, in_dims=(0, *in_dims))
            out = root_scalar(
                f_, *args, a=_a, b=_b, rtol=rtol, atol=atol, maxiter=maxiter
            )

            return out, 0

    return Root.apply(*args)
