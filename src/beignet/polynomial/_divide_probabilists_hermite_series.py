from torch import Tensor

from .__div import _div
from ._multiply_probabilists_hermite_series import multiply_probabilists_hermite_series


def divide_probabilists_hermite_series(
    input: Tensor,
    other: Tensor,
) -> (Tensor, Tensor):
    return _div(
        multiply_probabilists_hermite_series,
        input,
        other,
    )
