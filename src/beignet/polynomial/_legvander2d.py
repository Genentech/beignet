from torch import Tensor

from beignet.polynomial import legvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


def legvander2d(x: Tensor, y: Tensor, degree: Tensor) -> Tensor:
    return _flattened_vandermonde(
        (legvander, legvander),
        (x, y),
        degree,
    )
