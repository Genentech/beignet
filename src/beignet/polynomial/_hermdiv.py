from typing import Tuple

from torch import Tensor

from beignet.polynomial import hermmul
from beignet.polynomial.__div import _div


def hermdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(hermmul, input, other)
