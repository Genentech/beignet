from torch import Tensor

from beignet.polynomial import chebval


def chebgrid3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = chebval(arg, c)
    return c
