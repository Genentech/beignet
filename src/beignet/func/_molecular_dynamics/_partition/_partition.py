from typing import Callable, Optional

from torch import Tensor

from .__neighbor_list import _NeighborList
from .__neighbor_list_format import _NeighborListFormat
from ._neighbor_list import neighbor_list


def partition(
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    space: Tensor,
    neighborhood_radius: float,
    *,
    maximum_distance: float = 0.0,
    buffer_size_multiplier: float = 1.25,
    disable_unit_list: bool = False,
    mask_self: bool = True,
    mask_fn: Optional[Callable[[Tensor], Tensor]] = None,
    normalized: bool = False,
    neighbor_list_format: _NeighborListFormat = _NeighborListFormat.DENSE,
    **_,
) -> Callable[[Tensor, Optional[_NeighborList], Optional[int]], _NeighborList]:
    return neighbor_list(
        displacement_fn,
        space,
        neighborhood_radius,
        maximum_distance,
        buffer_size_multiplier,
        disable_unit_list,
        mask_self,
        mask_fn,
        normalized,
        neighbor_list_format,
    )
