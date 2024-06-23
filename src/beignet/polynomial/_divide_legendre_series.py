from torch import Tensor

from .__div import _div
from ._multiply_legendre_series import multiply_legendre_series


def divide_legendre_series(
    input: Tensor,
    other: Tensor,
) -> (Tensor, Tensor):
    return _div(
        multiply_legendre_series,
        input,
        other,
    )
