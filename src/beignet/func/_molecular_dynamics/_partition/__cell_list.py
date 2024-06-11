from typing import Callable, Dict

from torch import Tensor

from ..__dataclass import _dataclass
from .._static_field import static_field


@_dataclass
class _CellList:
    r"""Stores the spatial partition of a system into a cell list.

    See :meth:`cell_list` for details on the construction / specification.
    Cell list buffers all have a common shape, S, where
    * `S = [cell_count_x, cell_count_y, cell_capacity]`
    * `S = [cell_count_x, cell_count_y, cell_count_z, cell_capacity]`
    in two- and three-dimensions respectively. It is assumed that each cell has
    the same capacity.

    Attributes:
    positions_buffer: An ndarray of floating point positions with shape
      `S + [spatial_dimension]`.
    indexes: An ndarray of int32 particle ids of shape `S`. Note that empty
      slots are specified by `id = N` where `N` is the number of particles in
      the system.
    parameters: A dictionary of ndarrays of shape `S + [...]`. This contains
      side data placed into the cell list.
    exceeded_maximum_size: A boolean specifying whether or not the cell list
      exceeded the maximum allocated capacity.
    size: An integer specifying the maximum capacity of each cell in
      the cell list.
    item size: A tensor specifying the size of each cell in the cell list.
    update_fn: A function that updates the cell list at a fixed capacity.
    """
    exceeded_maximum_size: Tensor

    indexes: Tensor

    item_size: float = static_field()

    parameters: Dict[str, Tensor]

    positions_buffer: Tensor

    size: int = static_field()

    update_fn: Callable[..., "_CellList"] = static_field()

    def update(self, positions: Tensor, **kwargs) -> "_CellList":
        return self.update_fn(
            positions,
            [
                self.size,
                self.exceeded_maximum_size,
                self.update_fn,
            ],
            **kwargs,
        )
