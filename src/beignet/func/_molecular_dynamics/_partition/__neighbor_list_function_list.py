from typing import Callable

from torch import Tensor

from ..__dataclass import _dataclass
from .._static_field import static_field
from .__neighbor_list import _NeighborList


@_dataclass
class _NeighborListFunctionList:
    r"""
    A dataclass that encapsulates functions for setting up and updating a neighbor list.

    Attributes
    ----------
    setup_fn : Callable[..., _NeighborList]
        A function that sets up and returns a `_NeighborList` object.
    update_fn : Callable[[Tensor, _NeighborList], _NeighborList]
        A function that updates a `_NeighborList` object given a tensor and an existing `_NeighborList`.

    Methods
    -------
    __iter__()
        Returns an iterator over the setup and update functions.
    """

    setup_fn: Callable[..., _NeighborList] = static_field()

    update_fn: Callable[[Tensor, _NeighborList], _NeighborList] = static_field()

    def __iter__(self):
        return iter((self.setup_fn, self.update_fn))
