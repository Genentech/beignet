from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._chebvander import chebvander


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
