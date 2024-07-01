from typing import Tuple

from torch import Tensor

from beignet.polynomial import _div, legmul


def legdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(legmul, input, other)
