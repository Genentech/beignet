from torch import Tensor

from beignet.polynomial import _flattened_vandermonde, lagvander


def lagvander2d(
    x: Tensor,
    y: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (lagvander, lagvander),
        (x, y),
        degree,
    )
