from torch import Tensor

from .__div import _div
from ._multiply_laguerre_series import multiply_laguerre_series


def divide_laguerre_series(
    input: Tensor,
    other: Tensor,
) -> (Tensor, Tensor):
    return _div(
        multiply_laguerre_series,
        input,
        other,
    )
