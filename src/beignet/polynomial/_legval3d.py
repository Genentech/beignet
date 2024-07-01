from torch import Tensor

from beignet.polynomial import _evaluate, legval


def legval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(
        legval,
        c,
        x,
        y,
        z,
    )
