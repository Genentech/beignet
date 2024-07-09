from torch import Tensor

from ._evaluate_chebyshev_polynomial import evaluate_chebyshev_polynomial


def evaluate_chebyshev_polynomial_cartesian_3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = evaluate_chebyshev_polynomial(arg, c)
    return c
