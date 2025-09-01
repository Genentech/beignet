from torch import Tensor

from ._evaluate_polynomial import evaluate_polynomial


def evaluate_polynomial_cartesian_2d(
    x: Tensor, y: Tensor, coefficients: Tensor
) -> Tensor:
    r"""
    Parameters
    ----------
    x : Tensor

    y : Tensor

    coefficients : Tensor

    Returns
    -------
    output : Tensor
    """
    for input in [x, y]:
        coefficients = evaluate_polynomial(input, coefficients)

    return coefficients
