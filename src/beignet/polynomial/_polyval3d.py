from torch import Tensor

from beignet.polynomial import polyval
from beignet.polynomial.__evaluate import _evaluate


def polyval3d(x: Tensor, y: Tensor, z: Tensor, coefficients: Tensor) -> Tensor:
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
    return _evaluate(polyval, coefficients, x, y, z)
