from torch import Tensor

from beignet.polynomial import hermeline, hermemul
from beignet.polynomial.__from_roots import _from_roots


def hermefromroots(
    input: Tensor,
) -> Tensor:
    return _from_roots(hermeline, hermemul, input)
