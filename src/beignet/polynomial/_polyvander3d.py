from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._polyvander import polyvander


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
