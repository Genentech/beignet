from torch import Tensor

from beignet.polynomial import chebvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


def chebvander2d(
    x: Tensor,
    y: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (chebvander, chebvander),
        (x, y),
        degree,
    )
