import torch
from torch import Tensor


def laguerre_polynomial_companion(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    if input.shape[0] < 2:
        raise ValueError

    if input.shape[0] == 2:
        return torch.tensor([[1 + input[0] / input[1]]])

    n = input.shape[0] - 1

    output = torch.reshape(torch.zeros([n, n], dtype=input.dtype), [-1])

    output[1 :: n + 1] = -torch.arange(1, n)

    output[0 :: n + 1] = 2.0 * torch.arange(n) + 1.0

    output[n :: n + 1] = -torch.arange(1, n)

    output = torch.reshape(output, [n, n])

    output[:, -1] += (input[:-1] / input[-1]) * n

    return output
