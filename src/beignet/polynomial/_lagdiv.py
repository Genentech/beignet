from typing import Tuple

from torch import Tensor

from beignet.polynomial import lagmul
from beignet.polynomial.__div import _div


def lagdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(lagmul, input, other)
