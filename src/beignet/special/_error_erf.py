from torch import Tensor

from ._error_erfc import error_erfc


def error_erf(input: Tensor) -> Tensor:
    return 1.0 - error_erfc(input)
