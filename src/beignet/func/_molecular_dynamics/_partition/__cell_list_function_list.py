from typing import Callable

from torch import Tensor

from ..__dataclass import _dataclass
from .._static_field import static_field
from .__cell_list import _CellList


@_dataclass
class _CellListFunctionList:
    setup_fn: Callable[..., _CellList] = static_field()

    update_fn: Callable[[Tensor, _CellList | int], _CellList] = static_field()

    def __iter__(self):
        return iter([self.setup_fn, self.update_fn])
