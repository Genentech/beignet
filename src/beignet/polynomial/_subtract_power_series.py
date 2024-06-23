from torch import Tensor

from .__sub import _sub


def subtract_power_series(
    input: Tensor,
    other: Tensor,
) -> Tensor:
    return _sub(input, other)
