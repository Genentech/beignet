from torch import Tensor

from beignet.polynomial import _flattened_vandermonde, lagvander


def lagvander3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (lagvander, lagvander, lagvander),
        (x, y, z),
        degree,
    )
