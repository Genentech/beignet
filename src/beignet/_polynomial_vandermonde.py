import torch
from torch import Tensor


def polynomial_vandermonde(input: Tensor, degree: Tensor) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor

    degree : Tensor

    Returns
    -------
    output : Tensor
    """
    if degree < 0:
        raise ValueError

    degree = int(degree)

    input = torch.atleast_1d(input)

    output = torch.empty([degree + 1, *input.shape], dtype=input.dtype)

    output[0] = torch.ones_like(input)

    for i in range(1, degree + 1):
        output[i] = output[i - 1] * input

    output = torch.moveaxis(output, 0, -1)

    return output
