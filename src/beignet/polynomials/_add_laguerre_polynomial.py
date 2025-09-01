from torch import Tensor

from ._add_polynomial import add_polynomial


def add_laguerre_polynomial(input: Tensor, other: Tensor) -> Tensor:
    r"""
    Returns the sum of two polynomials.

    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    other : Tensor
        Polynomial coefficients.

    Returns
    -------
    output : Tensor
        Polynomial coefficients.
    """
    return add_polynomial(input, other)
