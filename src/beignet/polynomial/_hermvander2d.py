from torch import Tensor

from beignet.polynomial import hermvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


def hermvander2d(
    x: Tensor,
    y: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (hermvander, hermvander),
        (x, y),
        degree,
    )
