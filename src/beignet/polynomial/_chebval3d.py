from torch import Tensor

from .__evaluate import _evaluate
from ._chebval import chebval


def chebval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(chebval, c, x, y, z)
