from torch import Tensor

from .__pow import _pow
from ._lagmul import lagmul


def lagpow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        lagmul,
        input,
        exponent,
        maximum_exponent,
    )
