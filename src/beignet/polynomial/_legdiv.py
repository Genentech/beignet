from typing import Tuple

from torch import Tensor

from .__div import _div
from ._legmul import legmul


def legdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(legmul, input, other)
