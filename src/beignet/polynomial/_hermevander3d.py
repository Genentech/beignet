from torch import Tensor

from .__flattened_vandermonde import _flattened_vandermonde
from ._hermevander import hermevander


def hermevander3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (hermevander, hermevander, hermevander),
        (x, y, z),
        degree,
    )
