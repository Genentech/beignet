from torch import Tensor

from ._evaluate_polynomial import evaluate_polynomial


def evaluate_polynomial_3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    coefficients: Tensor,
) -> Tensor:
    r"""
    Parameters
    ----------
    x : Tensor

    y : Tensor

    z : Tensor

    coefficients : Tensor

    Returns
    -------
    output : Tensor
    """
    points = [x, y, z]

    if not all(a.shape == points[0].shape for a in points[1:]):
        match len(points):
            case 2:
                raise ValueError
            case 3:
                raise ValueError
            case _:
                raise ValueError

    points = iter(points)

    output = evaluate_polynomial(
        next(points),
        coefficients,
    )

    for x in points:
        output = evaluate_polynomial(
            x,
            output,
            tensor=False,
        )

    return output
