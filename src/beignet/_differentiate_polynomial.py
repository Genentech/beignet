import torch
from torch import Tensor


def differentiate_polynomial(
    input: Tensor,
    order: Tensor | None = None,
    scale: Tensor | None = None,
    dim: int = 0,
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
    input = torch.atleast_1d(input)

    if order == 0:
        return input

    input = torch.moveaxis(input, dim, 0)

    if order >= input.shape[0]:
        output = torch.zeros_like(input[:1])
    else:
        d = torch.arange(input.shape[0])

        output = input

        for _ in range(0, order):
            output = (d * output.T).T

            output = torch.roll(output, -1, dims=[0]) * scale

            output[-1] = 0.0

        output = output[:-order]

    output = torch.moveaxis(output, 0, dim)

    return output
