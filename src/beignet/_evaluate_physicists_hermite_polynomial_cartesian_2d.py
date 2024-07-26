from torch import Tensor

from ._evaluate_physicists_hermite_polynomial import (
    evaluate_physicists_hermite_polynomial,
)


def evaluate_physicists_hermite_polynomial_cartesian_2d(
    x: Tensor,
    y: Tensor,
    coefficients: Tensor,
) -> Tensor:
    for point in [x, y]:
        coefficients = evaluate_physicists_hermite_polynomial(
            point,
            coefficients,
        )

    return coefficients
