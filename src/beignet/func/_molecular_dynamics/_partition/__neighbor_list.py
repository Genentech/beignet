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
    CELL_LIST_OVERFLOW = 1 << 1
    CELL_SIZE_TOO_SMALL = 1 << 2
    MALFORMED_BOX = 1 << 3


PEC = PartitionErrorCode


@_dataclass
class _NeighborList:
    r"""A dataclass representing a neighbor list used in particle simulations.

    Attributes
    ----------
    buffer_fn : Callable[[Tensor, _CellList], _CellList]
        A function to buffer the cell list.
    indexes : Tensor
        A tensor containing the indexes of neighbors.
    item_size : float or None
        The size of each item in the neighbor list.
    maximum_size : int
        The maximum size of the neighbor list.
    format : _NeighborListFormat
        The format of the neighbor list.
    partition_error : _PartitionError
        An object representing partition errors.
    reference_positions : Tensor
        A tensor containing the reference positions of particles.
    units_buffer_size : int or None
        The buffer size in units.
    update_fn : Callable[[Tensor, "_NeighborList", Any], "_NeighborList"]
        A function to update the neighbor list.

    Methods
    -------
    update(positions: Tensor, **kwargs) -> "_NeighborList"
        Updates the neighbor list with new positions.
    did_buffer_overflow() -> bool
        Checks if the buffer overflowed.
    cell_size_too_small() -> bool
        Checks if the cell size is too small.
    malformed_box() -> bool
        Checks if the box is malformed.
    """

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
        return (
            self.partition_error.code
            & (PEC.NEIGHBOR_LIST_OVERFLOW | PEC.CELL_LIST_OVERFLOW)
        ).item() != 0

    @property
    def cell_size_too_small(self) -> bool:
        return (self.error.code & PEC.CELL_SIZE_TOO_SMALL).item() != 0

    @property
    def malformed_box(self) -> bool:
        return (self.error.code & PEC.MALFORMED_BOX).item() != 0
