from torch import Tensor

from beignet.polynomial import chebvander
from beignet.polynomial.__flattened_vandermonde import _flattened_vandermonde


def chebvander3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (chebvander, chebvander, chebvander),
        (x, y, z),
        degree,
    )
