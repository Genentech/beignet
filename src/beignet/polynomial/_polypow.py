from torch import Tensor

from beignet.polynomial import _pow, polymul


def polypow(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    exponent : float or Tensor

    maximum_exponent : float or Tensor, default=16.0

    Returns
    -------
    output : Tensor
        Polynomial coefficients of the power.
    """
    return _pow(
        polymul,
        input,
        exponent,
        maximum_exponent,
    )
