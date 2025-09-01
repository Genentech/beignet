from torch import Tensor

from ._evaluate_polynomial import evaluate_polynomial


def evaluate_polynomial_cartesian_3d(
    x: Tensor, y: Tensor, z: Tensor, coefficients: Tensor
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
    out : Tensor
    """
    for input in [x, y, z]:
        coefficients = evaluate_polynomial(
            input,
            coefficients,
        )

    return coefficients
