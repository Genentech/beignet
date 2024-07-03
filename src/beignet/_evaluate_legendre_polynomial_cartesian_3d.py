from torch import Tensor

from ._evaluate_legendre_polynomial import evaluate_legendre_polynomial


def evaluate_legendre_polynomial_cartesian_3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = evaluate_legendre_polynomial(arg, c)
    return c
