import torch
from torch import Tensor


def differentiate_probabilists_hermite_polynomial(
    input,
    order=1,
    scale=1,
    axis=0,
) -> Tensor:
    r"""
    Returns the derivative of a polynomial.

    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    order : Tensor, optional

    scale : Tensor, optional

    dim : int, default=0

    Returns
    -------
    output : Tensor
        Polynomial coefficients of the derivative.
    """
    if order < 0:
        raise ValueError

    input = torch.atleast_1d(input)

    if order == 0:
        return input

    input = torch.moveaxis(input, axis, 0)

    n = input.shape[0]

    if order >= n:
        input = torch.zeros_like(input[:1])
    else:
        for _ in range(order):
            n = n - 1

            input = input * scale

            der = torch.empty((n,) + input.shape[1:], dtype=input.dtype)

            j = torch.arange(n, 0, -1)

            der[j - 1] = (j * (input[j]).T).T

            input = der

    input = torch.moveaxis(input, 0, axis)

    return input
