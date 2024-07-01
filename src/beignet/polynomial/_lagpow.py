from torch import Tensor

from beignet.polynomial import lagmul
from beignet.polynomial.__pow import _pow


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
