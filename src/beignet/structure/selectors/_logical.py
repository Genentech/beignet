from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from .._residue_array import ResidueArray


@dataclass
class AndSelector:
    selector1: Callable | Tensor
    selector2: Callable | Tensor

    def __call__(self, input: ResidueArray, **kwargs):
        if callable(self.selector1):
            mask1 = self.selector1(input, **kwargs)
        else:
            mask1 = self.selector1

        if callable(self.selector2):
            mask2 = self.selector2(input, **kwargs)
        else:
            mask2 = self.selector2

        return mask1 & mask2
