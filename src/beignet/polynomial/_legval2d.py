from torch import Tensor

from .__evaluate import _evaluate
from ._legval import legval


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
