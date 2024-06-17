import torch
from torch import Tensor


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
    input = torch.atleast_1d(input)
    other = torch.atleast_1d(other)

    dtype = torch.promote_types(input.dtype, other.dtype)

    input = input.to(dtype)
    other = other.to(dtype)

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
