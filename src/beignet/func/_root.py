from typing import Callable

import torch
from torch import Tensor
from torch.autograd import Function


def root(
    func: Callable[[Tensor], Tensor],
    x0: Tensor,
    solve: Callable[[Callable, Tensor], Tensor],
    tangent_solve: Callable[[Callable, Tensor], Tensor],
    has_aux: bool = False,
) -> Tensor:
    r"""
    Differentiably solve for the roots of a function.

    Gradients are defined with respect to closed-over variables from the
    provided function, `func`, via the implicit function theorem.

    Parameters
    ----------
    func : Callable[[Tensor], Tensor]
        A Python function that takes one or more arguments.

    x0 : Tensor

    solve : Callable[[Callable, Tensor], Tensor]
        A Python function to solve for the roots of `func`. Must take two
        positional arguments, `func` and `x0`, and return a solution with the
        same structure as `x0` such that `func(solution) = 0`, i.e., the
        following is assumed:

        ```Python
        assert torch.all(func(solve(func, x0)) == 0.0)
        ```

    tangent_solve : Callable[[Callable, Tensor], Tensor]
        A Python function to solve the tangent system. Should take two
        positional arguments, a linear function (`func` linearized at its root)
        and a tensor, `y`, with the same structure as `x0`, returning a
        solution, `x`, such that `g(x) = y`:

        If `y` is a scalar, use:

        ```Python
        lambda g, y: y / g(1.0)
        ```

        If `y` is a vector, consider a linear solution with the Jacobian, if
        dimensionality of `y` is not too large:

        ```Python
        lambda g, y: torch.linalg.solve(jacobian(g)(y), y)
        ```

    has_aux : bool

    Returns
    -------
    output : Tensor
        The output of `solve(func, x0)` with gradients defined via implicit
        differentiation assuming `func(solve(func, x0)) == 0`.
    """

    class Root(Function):
        @staticmethod
        def forward(ctx, initial_guess) -> Tensor:
            if has_aux:
                solution, aux = solve(func, initial_guess)

                ctx.aux = aux
            else:
                solution = solve(func, initial_guess)

            ctx.save_for_backward(solution)

            return solution

        @staticmethod
        def backward(ctx, grad_output) -> Tensor:
            (solution,) = ctx.saved_tensors

            _, jvp_fn = torch.func.linearize(func, solution)

            def f(x):
                return jvp_fn(x)

            return tangent_solve(f, grad_output)

    return Root.apply(x0)
