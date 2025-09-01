from torch import Tensor

from ._evaluate_physicists_hermite_polynomial import (
    evaluate_physicists_hermite_polynomial,
)


def evaluate_physicists_hermite_polynomial_cartesian_3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = evaluate_physicists_hermite_polynomial(arg, c)
    return c
