from torch import Tensor

from ._evaluate_probabilists_hermite_polynomial import (
    evaluate_probabilists_hermite_polynomial,
)


def evaluate_probabilists_hermite_polynomial_cartersian_2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = evaluate_probabilists_hermite_polynomial(arg, c)
    return c
