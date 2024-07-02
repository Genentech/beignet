from torch import Tensor

from .__evaluate import _evaluate
from ._legval import legval


def legval3d(x: Tensor, y: Tensor, z: Tensor, c: Tensor) -> Tensor:
    return _evaluate(legval, c, x, y, z)
