from torch import Tensor

from .__pow import _pow
from ._hermemul import hermemul


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
