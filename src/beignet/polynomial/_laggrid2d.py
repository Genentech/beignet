from torch import Tensor

from ._lagval import lagval


def laggrid2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = lagval(arg, c)
    return c
