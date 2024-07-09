from torch import Tensor

from ._evaluate_chebyshev_polynomial import evaluate_chebyshev_polynomial


def evaluate_chebyshev_polynomial_cartesian_2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = evaluate_chebyshev_polynomial(arg, c)
    return c
