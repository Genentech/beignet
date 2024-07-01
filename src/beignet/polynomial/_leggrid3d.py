from torch import Tensor

from beignet.polynomial import legval


def leggrid3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = legval(arg, c)
    return c
