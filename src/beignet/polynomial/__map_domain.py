from torch import Tensor

from .__map_parameters import _map_parameters


def _map_domain(x: Tensor, previous: Tensor, new: Tensor) -> Tensor:
    off, scale = _map_parameters(previous, new)

    return off + scale * x
