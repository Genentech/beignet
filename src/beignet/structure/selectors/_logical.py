import typing
from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from .._invoke_selector import invoke_selector

if typing.TYPE_CHECKING:
    from .. import ResidueArray


@dataclass
class AndSelector:
    selector1: Callable | Tensor | None
    selector2: Callable | Tensor | None

    def __call__(self, input: "ResidueArray", **kwargs):
        mask1 = invoke_selector(self.selector1, input, **kwargs)
        mask2 = invoke_selector(self.selector2, input, **kwargs)

        return mask1 & mask2
