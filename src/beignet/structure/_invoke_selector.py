import typing
from typing import Callable

import torch
from torch import Tensor

if typing.TYPE_CHECKING:
    from . import ResidueArray


def invoke_selector(
    selector: Callable[["ResidueArray"], Tensor] | Tensor | None,
    input: "ResidueArray",
    **kwargs,
):
    if callable(selector):
        mask = selector(input, **kwargs)
    elif isinstance(selector, Tensor):
        mask = selector
    elif selector is None:
        mask = torch.ones_like(input.atom_thin_mask)
    else:
        raise AssertionError(f"{selector=} not supported")

    return mask
