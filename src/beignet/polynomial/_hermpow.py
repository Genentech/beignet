from torch import Tensor

from beignet.polynomial import hermmul
from beignet.polynomial.__pow import _pow


def hermpow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    return _pow(
        hermmul,
        input,
        exponent,
        maximum_exponent,
    )
