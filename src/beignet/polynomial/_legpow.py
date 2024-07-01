from torch import Tensor

from beignet.polynomial import _pow, legmul


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
