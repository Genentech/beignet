from torch import Tensor

from beignet.polynomial import _from_roots, polyline, polymul


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
