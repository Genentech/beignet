from torch import Tensor

from .__from_roots import _from_roots
from ._chebline import chebline
from ._chebmul import chebmul


def chebfromroots(
    input: Tensor,
) -> Tensor:
    return _from_roots(chebline, chebmul, input)
