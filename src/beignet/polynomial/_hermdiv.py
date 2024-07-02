from typing import Tuple

from torch import Tensor

from .__div import _div
from ._hermmul import hermmul


def hermdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(hermmul, input, other)
