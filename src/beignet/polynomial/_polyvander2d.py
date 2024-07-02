from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._polyvander import polyvander


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
