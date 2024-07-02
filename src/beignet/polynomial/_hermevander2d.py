from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._hermevander import hermevander


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
