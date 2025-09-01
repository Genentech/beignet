import torch
from torch import Tensor


def trim_polynomial_coefficients(
    input: Tensor,
    tol: float = 0.0,
) -> Tensor:
    if tol < 0:
        raise ValueError

    input = torch.atleast_1d(input)

    indices = torch.nonzero(torch.abs(input) > tol)

    if indices.shape[0] == 0:
        output = input[:1] * 0
    else:
        output = input[: indices[-1] + 1]

    return output
