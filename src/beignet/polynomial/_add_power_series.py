from torch import Tensor

from .__add import _add


def add_power_series(input: Tensor, other: Tensor) -> Tensor:
    return _add(input, other)
