from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._hermvander import hermvander


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
