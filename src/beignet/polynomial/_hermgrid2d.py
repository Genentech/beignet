from torch import Tensor

from ._hermval import hermval


def hermgrid2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = hermval(arg, c)
    return c
