from torch import Tensor

from beignet.polynomial import legvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


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
