from torch import Tensor

from ._legval import legval


def leggrid2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = legval(arg, c)
    return c
