from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._legvander import legvander


def legvander3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (legvander, legvander, legvander),
        (x, y, z),
        degree,
    )
