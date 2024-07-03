import torch
from torch import Tensor


def evaluate_polynomial(
    input: Tensor, coefficients: Tensor, tensor: bool = True
) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor

    coefficients : Tensor

    tensor : bool

    Returns
    -------
    output : Tensor
    """
    coefficients = torch.atleast_1d(coefficients)

    if tensor:
        coefficients = torch.reshape(
            coefficients,
            coefficients.shape + (1,) * input.ndim,
        )

    output = coefficients[-1] + torch.zeros_like(input)

    for i in range(2, coefficients.shape[0] + 1):
        output = coefficients[-i] + output * input

    return output
