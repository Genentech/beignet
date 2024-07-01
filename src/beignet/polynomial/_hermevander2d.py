from torch import Tensor

from beignet.polynomial import hermevander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


def hermevander2d(
    x: Tensor,
    y: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (hermevander, hermevander),
        (x, y),
        degree,
    )
