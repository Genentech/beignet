import torch
from torch import Tensor

from .__as_series import _as_series


def polycompanion(input: Tensor) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor
        Polynomial coefficients.

    Returns
    -------
    output : Tensor, shape=(degree, degree)
        Companion matrix.
    """
    [input] = _as_series([input])

    if input.shape[0] < 2:
        raise ValueError

    if input.shape[0] == 2:
        return torch.tensor([[-input[0] / input[1]]])

    n = input.shape[0] - 1

    output = torch.reshape(torch.zeros([n, n], dtype=input.dtype), [-1])

    output[n :: n + 1] = 1.0

    output = torch.reshape(output, [n, n])

    output[:, -1] = output[:, -1] + (-input[:-1] / input[-1])

    return output
