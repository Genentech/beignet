from torch import Tensor

from beignet.polynomial import chebline, chebmul
from beignet.polynomial.__from_roots import _from_roots


def chebfromroots(
    input: Tensor,
) -> Tensor:
    return _from_roots(chebline, chebmul, input)
