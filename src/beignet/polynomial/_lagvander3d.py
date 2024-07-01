from torch import Tensor

from beignet.polynomial import lagvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


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
