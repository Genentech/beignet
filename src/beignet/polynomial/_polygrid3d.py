from torch import Tensor

from beignet.polynomial import polyval


def polygrid3d(x: Tensor, y: Tensor, z: Tensor, coefficients: Tensor) -> Tensor:
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
        coefficients = polyval(
            input,
            coefficients,
        )

    return coefficients
