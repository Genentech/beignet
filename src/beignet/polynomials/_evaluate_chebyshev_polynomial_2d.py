from torch import Tensor

from ._evaluate_chebyshev_polynomial import evaluate_chebyshev_polynomial


def evaluate_chebyshev_polynomial_2d(
    x: Tensor,
    y: Tensor,
    coefficients: Tensor,
) -> Tensor:
    points = [x, y]

    if not all(a.shape == points[0].shape for a in points[1:]):
        match len(points):
            case 2:
                raise ValueError
            case 3:
                raise ValueError
            case _:
                raise ValueError

    points = iter(points)

    output = evaluate_chebyshev_polynomial(
        next(points),
        coefficients,
    )

    for x in points:
        output = evaluate_chebyshev_polynomial(
            x,
            output,
            tensor=False,
        )

    return output
