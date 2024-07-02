from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._chebvander import chebvander


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
