from torch import Tensor

from .__pow import _pow
from ._multiply_legendre_series import multiply_legendre_series


def pow_legendre_series(input: Tensor, exponent, maximum_exponent=16) -> Tensor:
    return _pow(multiply_legendre_series, input, exponent, maximum_exponent)
