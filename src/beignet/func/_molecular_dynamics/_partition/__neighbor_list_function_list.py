from typing import Callable

from torch import Tensor

from ..__dataclass import _dataclass
from .._static_field import static_field
from .__neighbor_list import _NeighborList


@_dataclass
class _NeighborListFunctionList:
    setup_fn: Callable[..., _NeighborList] = static_field()

    update_fn: Callable[[Tensor, _NeighborList], _NeighborList] = static_field()

    def __iter__(self):
        return iter((self.setup_fn, self.update_fn))
