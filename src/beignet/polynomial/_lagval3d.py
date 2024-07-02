from torch import Tensor

from .__evaluate import _evaluate
from ._lagval import lagval


def lagval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(lagval, c, x, y, z)
