from torch import Tensor

from beignet.polynomial import _flattened_vandermonde, hermevander


def hermevander2d(
    x: Tensor,
    y: Tensor,
    degree: Tensor,
) -> Tensor:
    return _flattened_vandermonde(
        (hermevander, hermevander),
        (x, y),
        degree,
    )
