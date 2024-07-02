from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._lagvander import lagvander


def lagvander3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (lagvander, lagvander, lagvander),
        (x, y, z),
        degree,
    )
