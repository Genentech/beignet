import torch
from torch import Tensor


def polynomial_roots(input: Tensor) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    Returns
    -------
    output : Tensor
        Roots.
    """
    input = torch.atleast_1d(input)

    if input.shape[0] < 2:
        return torch.tensor([], dtype=input.dtype)

    if input.shape[0] == 2:
        return torch.tensor([-input[0] / input[1]])

    if input.shape[0] < 2:
        raise ValueError

    if input.shape[0] == 2:
        output = torch.tensor([[-input[0] / input[1]]])
    else:
        n = input.shape[0] - 1

        output = torch.reshape(torch.zeros([n, n], dtype=input.dtype), [-1])

        output[n :: n + 1] = 1.0

        output = torch.reshape(output, [n, n])

        output[:, -1] = output[:, -1] + (-input[:-1] / input[-1])

    output = torch.flip(output, dims=[0])
    output = torch.flip(output, dims=[1])

    output = torch.linalg.eigvals(output)

    output, _ = torch.sort(output.real)

    return output
