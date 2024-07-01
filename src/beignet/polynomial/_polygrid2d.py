from torch import Tensor

from beignet.polynomial import polyval


def polygrid2d(x: Tensor, y: Tensor, coefficients: Tensor) -> Tensor:
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
        coefficients = polyval(input, coefficients)

    return coefficients
