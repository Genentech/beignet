from typing import Tuple

from torch import Tensor

from .__div import _div
from ._hermemul import hermemul


def hermediv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(hermemul, input, other)
