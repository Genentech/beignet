from torch import Tensor

from beignet.polynomial import _flattened_vandermonde, legvander


def legvander2d(
    x: Tensor,
    y: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (legvander, legvander),
        (x, y),
        degree,
    )
