from typing import Literal

import torchaudio.functional
from torch import Tensor

from .__as_series import _as_series


def polymul(
    input: Tensor,
    other: Tensor,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    other : Tensor
        Polynomial coefficients.

    mode : Literal["full", "same", "valid"]

    Returns
    -------
    output : Tensor
        Polynomial coefficients of the product.
    """
    [input, other] = _as_series([input, other])

    output = torchaudio.functional.convolve(input, other)

    if mode == "same":
        output = output[: max(input.shape[0], other.shape[0])]

    return output
