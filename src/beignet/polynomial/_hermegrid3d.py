from torch import Tensor

from beignet.polynomial import hermeval


def hermegrid3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    for arg in [x, y, z]:
        c = hermeval(arg, c)
    return c
