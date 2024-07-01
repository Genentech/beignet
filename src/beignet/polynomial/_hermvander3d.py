from torch import Tensor

from beignet.polynomial import _flattened_vandermonde, hermvander


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
