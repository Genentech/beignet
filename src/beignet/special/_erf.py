from torch import Tensor

from ._erfc import erfc


def erf(input: Tensor) -> Tensor:
    return 1.0 - erfc(input)
