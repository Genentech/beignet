from torch import Tensor

from beignet.polynomial import _flattened_vandermonde, polyvander


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
