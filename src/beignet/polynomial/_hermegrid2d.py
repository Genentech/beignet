from torch import Tensor

from ._hermeval import hermeval


def hermegrid2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y]:
        c = hermeval(arg, c)
    return c
