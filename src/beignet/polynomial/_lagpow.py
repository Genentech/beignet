from torch import Tensor

from beignet.polynomial import _pow, lagmul


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
