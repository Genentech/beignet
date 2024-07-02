from typing import Tuple

from torch import Tensor

from .__div import _div
from ._lagmul import lagmul


def lagdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(lagmul, input, other)
