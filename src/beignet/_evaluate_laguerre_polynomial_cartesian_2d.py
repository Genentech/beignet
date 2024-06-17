from torch import Tensor

from ._evaluate_laguerre_polynomial import evaluate_laguerre_polynomial


def evaluate_laguerre_polynomial_cartesian_2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = evaluate_laguerre_polynomial(arg, c)
    return c
