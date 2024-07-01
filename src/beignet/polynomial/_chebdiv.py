from typing import Tuple

from torch import Tensor

from beignet.polynomial import chebmul
from beignet.polynomial.__div import _div


def chebdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(chebmul, input, other)
