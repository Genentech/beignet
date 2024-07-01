from typing import Tuple

from torch import Tensor

from beignet.polynomial import polymul

from .__as_series import _as_series
from .__div import _div


def polydiv(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]:
    r"""
    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    other : Tensor
        Polynomial coefficients.

    Returns
    -------
    output : Tuple[Tensor, Tensor]
        Polynomial coefficients of the quotient and remainder.
    """
    [input, other] = _as_series([input, other])

    return _div(polymul, input, other)
