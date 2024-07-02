from typing import Tuple

from torch import Tensor

from .__div import _div
from ._chebmul import chebmul


def chebdiv(
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _div(chebmul, input, other)
