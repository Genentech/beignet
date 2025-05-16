import typing
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .._invoke_selector import invoke_selector

if typing.TYPE_CHECKING:
    from .. import ResidueArray


@dataclass
class AndSelector:
    selectors: list[Callable | Tensor | None]

    def __call__(self, input: "ResidueArray", **kwargs):
        mask = torch.ones_like(input.atom_thin_mask)
        for selector in self.selectors:
            if selector is not None:
                mask = mask & invoke_selector(selector, input, **kwargs)

        return mask


@dataclass
class OrSelector:
    selectors: list[Callable | Tensor | None]

    def __call__(self, input: "ResidueArray", **kwargs):
        mask = torch.zeros_like(input.atom_thin_mask)
        for selector in self.selectors:
            if selector is not None:
                mask = mask | invoke_selector(selector, input, **kwargs)

        return mask


@dataclass
class NotSelector:
    selector: Callable | Tensor

    def __call__(self, input: "ResidueArray", **kwargs):
        mask = invoke_selector(self.selector, input, **kwargs)
        return ~mask
