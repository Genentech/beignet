from torch import Tensor

from .__pow import _pow
from ._legmul import legmul


def legpow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        legmul,
        input,
        exponent,
        maximum_exponent,
    )
