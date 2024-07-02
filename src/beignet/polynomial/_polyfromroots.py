from torch import Tensor

from .__from_roots import _from_roots
from ._polyline import polyline
from ._polymul import polymul


def polyfromroots(input: Tensor) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor
        Roots.

    Returns
    -------
    output : Tensor
        Polynomial coefficients.
    """
    return _from_roots(polyline, polymul, input)
