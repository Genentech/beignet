import torch
from torch import Tensor


def add_polynomial(input: Tensor, other: Tensor) -> Tensor:
    r"""
    Returns the sum of two polynomials.

    Parameters
    ----------
    input : Tensor
        Polynomial coefficients as a 0D or 1D tensor.

    other : Tensor
        Polynomial coefficients as a 0D or 1D tensor.

    Returns
    -------
    output : Tensor
        Coefficients of the sum polynomial as a 1D tensor.

    Notes
    -----
    If inputs have differing `dtype`, the output will be promoted to a common `dtype`.
    Inputs of differing lengths are supported, with all trailing coefficients of both inputs assumed to be zero.

    Examples
    --------
    >>> left = torch.tensor([1, 2])
    >>> right = torch.tensor(0.5)
    >>> add_polynomial(left, right)
    tensor([1.5000, 2.0000])
    """
    input = torch.atleast_1d(input)
    other = torch.atleast_1d(other)

    if input.ndim > 1 or other.ndim > 1:
        raise ValueError(
            f"Inputs may not be more than 1D. Got inputs of shape {tuple(input.shape)} and {tuple(other.shape)}."
        )

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
