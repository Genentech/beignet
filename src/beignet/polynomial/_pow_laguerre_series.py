from torch import Tensor

from .__pow import _pow
from ._multiply_laguerre_series import multiply_laguerre_series


def pow_laguerre_series(
    input: Tensor,
    exponent: Tensor,
    maximum_exponent: Tensor = 16,
) -> Tensor:
    return _pow(
        multiply_laguerre_series,
        input,
        exponent,
        maximum_exponent,
    )
