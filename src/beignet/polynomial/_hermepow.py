from torch import Tensor

from beignet.polynomial import hermemul
from beignet.polynomial.__pow import _pow


def hermepow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        hermemul,
        input,
        exponent,
        maximum_exponent,
    )
