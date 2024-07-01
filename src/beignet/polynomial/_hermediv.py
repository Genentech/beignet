from typing import Tuple

from torch import Tensor

from beignet.polynomial import hermemul
from beignet.polynomial.__div import _div


def hermediv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(hermemul, input, other)
