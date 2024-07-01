from torch import Tensor

from beignet.polynomial import _flattened_vandermonde, hermevander


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
