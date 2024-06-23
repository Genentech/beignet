from torch import Tensor

from .__div import _div
from ._multiply_physicists_hermite_series import multiply_physicists_hermite_series


def divide_physicists_hermite_series(
    input: Tensor,
    other: Tensor,
) -> (Tensor, Tensor):
    return _div(
        multiply_physicists_hermite_series,
        input,
        other,
    )
