import torch
from torch import Tensor


def subtract_laguerre_polynomial(input: Tensor, other: Tensor) -> Tensor:
    r"""
    Returns the difference of two polynomials.

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
    input = torch.atleast_1d(input)
    other = torch.atleast_1d(other)

    dtype = torch.promote_types(input.dtype, other.dtype)

    input = input.to(dtype)
    other = other.to(dtype)

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
