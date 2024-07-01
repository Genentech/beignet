import torch
from torch import Tensor

from .__as_series import _as_series


def lagsub(input: Tensor, other: Tensor) -> Tensor:
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
        Polynomial coefficients of the difference.
    """
    [input, other] = _as_series([input, other])

    if input.shape[0] > other.shape[0]:
        output = -other

        output = torch.concatenate(
            [
                output,
                torch.zeros(
                    input.shape[0] - other.shape[0],
                    dtype=other.dtype,
                ),
            ],
        )
        output = input + output
    else:
        output = -other

        output = torch.concatenate(
            [
                output[: input.shape[0]] + input,
                output[input.shape[0] :],
            ],
        )

    return output
