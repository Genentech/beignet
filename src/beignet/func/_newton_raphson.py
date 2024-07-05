import math

import torch
import torch.autograd.functional
import torch.linalg


def newton_raphson(func, x0):
    tolerance = 1e-16

    maximum_iterations = 20

    iteration = 0

    x = x0

    y = func(x0)

    jacobian = torch.autograd.functional.jacobian(func, x0)

    while (torch.max(torch.abs(y)) > tolerance) & (iteration < maximum_iterations):
        step = torch.reshape(
            torch.linalg.solve(
                torch.reshape(
                    jacobian,
                    [-1, math.prod(y.shape)],
                ),
                torch.ravel(y),
            ),
            y.shape,
        )

        x = x - step

        y = func(x)

        jacobian = torch.autograd.functional.jacobian(func, x)

        iteration = iteration + 1

    return x
