from torch import Tensor

from ._hermval import hermval


def hermgrid3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = hermval(arg, c)
    return c
