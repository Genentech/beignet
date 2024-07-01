from torch import Tensor

from beignet.polynomial import polyvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


def polyvander3d(x: Tensor, y: Tensor, z: Tensor, degree: Tensor) -> Tensor:
    r"""
    Parameters
    ----------
    x : Tensor

    y : Tensor

    z : Tensor

    degree : Tensor

    Returns
    -------
    output : Tensor
    """
    return _flattened_vandermonde(
        (polyvander, polyvander, polyvander),
        (x, y, z),
        degree,
    )
