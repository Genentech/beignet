import torch
from torch import Tensor

from .__as_series import _as_series


def polyadd(input: Tensor, other: Tensor) -> Tensor:
    r"""
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
    [input] = _as_series([input])

    if input.shape[0] > other.shape[0]:
        output = torch.concatenate(
            [
                other,
                torch.zeros(
                    input.shape[0] - other.shape[0],
                    dtype=other.dtype,
                ),
            ],
        )

        output = input + output
    else:
        output = torch.concatenate(
            [
                input,
                torch.zeros(
                    other.shape[0] - input.shape[0],
                    dtype=input.dtype,
                ),
            ]
        )

        output = other + output

    return output
