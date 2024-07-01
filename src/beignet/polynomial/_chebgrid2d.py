from torch import Tensor

from beignet.polynomial import chebval


def chebgrid2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = chebval(arg, c)
    return c
