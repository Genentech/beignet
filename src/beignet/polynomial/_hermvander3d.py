from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._hermvander import hermvander


def hermvander3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (hermvander, hermvander, hermvander),
        (x, y, z),
        degree,
    )
