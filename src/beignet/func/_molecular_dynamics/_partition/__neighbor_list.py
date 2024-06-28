from enum import IntEnum
from typing import Any, Callable
from torch import Tensor

from ..__dataclass import _dataclass
from .._static_field import static_field
from .__cell_list import _CellList
from .__neighbor_list_format import _NeighborListFormat
from .__partition_error import _PartitionError


class PartitionErrorCode(IntEnum):
  """An enum specifying different error codes.

  Attributes:
    NONE: Means that no error was encountered during simulation.
    NEIGHBOR_LIST_OVERFLOW: Indicates that the neighbor list was not large
      enough to contain all of the particles. This should indicate that it is
      necessary to allocate a new neighbor list.
    CELL_LIST_OVERFLOW: Indicates that the cell list was not large enough to
      contain all of the particles. This should indicate that it is necessary
      to allocate a new cell list.
    CELL_SIZE_TOO_SMALL: Indicates that the size of cells in a cell list was
      not large enough to properly capture particle interactions. This
      indicates that it is necessary to allcoate a new cell list with larger
      cells.
    MALFORMED_BOX: Indicates that a box matrix was not properly upper
      triangular.
  """
  NONE = 0
  NEIGHBOR_LIST_OVERFLOW = 1 << 0
  CELL_LIST_OVERFLOW     = 1 << 1
  CELL_SIZE_TOO_SMALL    = 1 << 2
  MALFORMED_BOX          = 1 << 3
PEC = PartitionErrorCode


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

    @property
    def did_buffer_overflow(self) -> bool:
        return (self.partition_error.code & (PEC.NEIGHBOR_LIST_OVERFLOW | PEC.CELL_LIST_OVERFLOW)).item() != 0
