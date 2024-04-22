from typing import Any, Callable

from torch import Tensor

from ..__dataclass import _dataclass
from .._static_field import static_field
from .__cell_list import _CellList
from .__neighbor_list_format import _NeighborListFormat
from .__partition_error import _PartitionError


@_dataclass
class _NeighborList:
    buffer_fn: Callable[[Tensor, _CellList], _CellList] = static_field()

    indexes: Tensor

    item_size: float | None = static_field()

    maximum_size: int = static_field()

    format: _NeighborListFormat = static_field()

    partition_error: _PartitionError

    reference_positions: Tensor

    units_buffer_size: int | None = static_field()

    update_fn: Callable[[Tensor, "_NeighborList", Any], "_NeighborList"] = (
        static_field()
    )

    def update(self, positions: Tensor, **kwargs) -> "_NeighborList":
        return self.update_fn(positions, self, **kwargs)
