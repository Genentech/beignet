from torch import Tensor

from .__evaluate import _evaluate
from ._hermeval import hermeval


def hermeval3d(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    c: Tensor,
) -> Tensor:
    return _evaluate(hermeval, c, x, y, z)
