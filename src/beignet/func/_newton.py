from typing import Callable

import torch.linalg
from torch import Tensor
from torch.autograd.functional import jacobian

from ._root import root


def newton(
    func: Callable[[Tensor], Tensor],
    x0,
    maximum_iterations: int = 50,
    tolerance: float = 1.48e-08,
) -> Tensor:
    def solve(f, x):
        for _ in range(maximum_iterations):
            fx = f(x)

            if torch.max(torch.abs(fx)) <= tolerance:
                break

            step = torch.linalg.solve(
                torch.reshape(
                    jacobian(f, x),
                    [-1, torch.numel(fx)],
                ),
                torch.reshape(
                    fx,
                    [-1, 1],
                ),
            )

            x = torch.reshape(
                x - step,
                x.shape,
            )

        return x

    def tangent_solve(g, y):
        return torch.reshape(
            torch.linalg.solve(
                torch.reshape(jacobian(g, y), [-1, *torch.numel(y)]),
                torch.reshape(y, [-1, 1]),
            ),
            y.shape,
        )

    return root(func, x0, solve, tangent_solve)
