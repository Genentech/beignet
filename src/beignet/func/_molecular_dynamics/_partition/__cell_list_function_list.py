from typing import Callable

from torch import Tensor

from ..__dataclass import _dataclass
from .._static_field import static_field
from .__cell_list import _CellList


@_dataclass
class _CellListFunctionList:
    r"""A dataclass that encapsulates functions for setting up and updating a cell list.

    Attributes
    ----------
    setup_fn : Callable[..., _CellList]
        A function that sets up and returns a `_CellList` object.
    update_fn : Callable[[Tensor, Union[_CellList, int]], _CellList]
        A function that updates a `_CellList` object given a tensor and either an existing `_CellList` or an integer.

    Methods
    -------
    __iter__()
        Returns an iterator over the setup and update functions.
    """

    setup_fn: Callable[..., _CellList] = static_field()

    update_fn: Callable[[Tensor, _CellList | int], _CellList] = static_field()

    def __iter__(self):
        return iter([self.setup_fn, self.update_fn])
