from torch import Tensor

from beignet.polynomial import _pow, hermmul


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
