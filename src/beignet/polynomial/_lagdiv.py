from typing import Tuple

from torch import Tensor

from beignet.polynomial import _div, lagmul


def lagdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(lagmul, input, other)
