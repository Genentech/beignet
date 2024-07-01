from torch import Tensor

from beignet.polynomial import _from_roots, hermeline, hermemul


def hermefromroots(
    input: Tensor,
) -> Tensor:
    return _from_roots(hermeline, hermemul, input)
