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
    r"""Partitions the space into a neighbor list based on the given parameters.

    Parameters
    ----------
    displacement_fn : Callable[[Tensor, Tensor], Tensor]
        A function that computes the displacement between two tensors.
    space : Tensor
        The space in which the particles are located.
    neighborhood_radius : float
        The radius within which neighbors are considered.
    maximum_distance : float, optional
        The maximum distance for considering neighbors (default is 0.0).
    buffer_size_multiplier : float, optional
        A multiplier for the buffer size (default is 1.25).
    disable_unit_list : bool, optional
        Whether to disable the unit list (default is False).
    mask_self : bool, optional
        Whether to mask self interactions (default is True).
    mask_fn : Optional[Callable[[Tensor], Tensor]], optional
        A function to mask certain interactions (default is None).
    normalized : bool, optional
        Whether the space is normalized (default is False).
    neighbor_list_format : _NeighborListFormat, optional
        The format of the neighbor list (default is _NeighborListFormat.DENSE).

    Returns
    -------
    Callable[[Tensor, Optional[_NeighborList], Optional[int]], _NeighborList]
        A function that generates a neighbor list based on the given parameters.
    """
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
