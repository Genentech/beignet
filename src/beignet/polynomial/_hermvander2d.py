from torch import Tensor

from beignet.polynomial import _flattened_vandermonde, hermvander


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
