from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._lagvander import lagvander


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
