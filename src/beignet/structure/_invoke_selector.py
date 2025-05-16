import typing
from typing import Callable

from torch import Tensor

if typing.TYPE_CHECKING:
    from ._residue_array import ResidueArray


def invoke_selector(
    selector: Callable[["ResidueArray"], Tensor] | Tensor,
    input: "ResidueArray",
    **kwargs,
):
    if callable(selector):
        mask = selector(input, **kwargs)
    elif isinstance(selector, Tensor):
        mask = selector
    else:
        raise AssertionError(f"{selector=} not supported")

    return mask
