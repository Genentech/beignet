from torch import Tensor

from beignet.polynomial import legval


def leggrid2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = legval(arg, c)
    return c
