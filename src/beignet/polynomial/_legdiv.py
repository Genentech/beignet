from typing import Tuple

from torch import Tensor

from beignet.polynomial import legmul
from beignet.polynomial.__div import _div


def legdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(legmul, input, other)
