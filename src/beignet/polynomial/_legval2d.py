from torch import Tensor

from beignet.polynomial import _evaluate, legval


def legval2d(
    x: Tensor,
    y: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(
        legval,
        c,
        x,
        y,
    )
