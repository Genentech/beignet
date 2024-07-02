from torch import Tensor

from .__evaluate import _evaluate
from ._polyval import polyval


def polyval2d(x: Tensor, y: Tensor, coefficients: Tensor) -> Tensor:
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
    return _evaluate(polyval, coefficients, x, y)
