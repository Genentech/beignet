import torch
from torch import Tensor


def legendre_polynomial_companion(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    if input.shape[0] < 2:
        raise ValueError

    if input.shape[0] == 2:
        return torch.tensor([[-input[0] / input[1]]])

    n = input.shape[0] - 1

    output = torch.zeros((n, n), dtype=input.dtype)

    scale = 1.0 / torch.sqrt(2 * torch.arange(n) + 1)

    shape = output.shape

    output = torch.reshape(output, [-1])

    output[1 :: n + 1] = torch.arange(1, n) * scale[: n - 1] * scale[1:n]

    output[n :: n + 1] = torch.arange(1, n) * scale[: n - 1] * scale[1:n]

    output = torch.reshape(output, shape)

    output[:, -1] += -(input[:-1] / input[-1]) * (scale / scale[-1]) * (n / (2 * n - 1))

    return output
