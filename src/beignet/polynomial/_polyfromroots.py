from torch import Tensor

from beignet.polynomial import polyline, polymul
from beignet.polynomial.__from_roots import _from_roots


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
