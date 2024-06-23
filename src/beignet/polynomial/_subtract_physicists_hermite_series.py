from torch import Tensor

from .__sub import _sub


def subtract_physicists_hermite_series(
    input: Tensor,
    other: Tensor,
) -> Tensor:
    return _sub(input, other)
