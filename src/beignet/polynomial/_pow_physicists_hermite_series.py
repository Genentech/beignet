from torch import Tensor

from .__pow import _pow
from ._multiply_physicists_hermite_series import multiply_physicists_hermite_series


def pow_physicists_hermite_series(
    input: Tensor, exponent, maximum_exponent=16
) -> Tensor:
    return _pow(
        multiply_physicists_hermite_series,
        input,
        exponent,
        maximum_exponent,
    )
