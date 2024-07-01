from torch import Tensor

from beignet.polynomial import lagvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


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
