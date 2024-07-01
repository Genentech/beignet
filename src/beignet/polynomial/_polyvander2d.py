from torch import Tensor

from beignet.polynomial import polyvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


def polyvander2d(x: Tensor, y: Tensor, degree: Tensor) -> Tensor:
    r"""
    Parameters
    ----------
    x : Tensor

    y : Tensor

    degree : Tensor

    Returns
    -------
    output : Tensor
    """
    return _flattened_vandermonde(
        (polyvander, polyvander),
        (x, y),
        degree,
    )
