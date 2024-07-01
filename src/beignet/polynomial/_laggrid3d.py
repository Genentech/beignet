from torch import Tensor

from beignet.polynomial import lagval


def laggrid3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = lagval(arg, c)
    return c
