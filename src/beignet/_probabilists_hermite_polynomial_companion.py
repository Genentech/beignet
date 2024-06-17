import torch
from torch import Tensor


def probabilists_hermite_polynomial_companion(input: Tensor) -> Tensor:
    input = torch.atleast_1d(input)

    if input.shape[0] < 2:
        raise ValueError

    if input.shape[0] == 2:
        return torch.tensor([[-input[0] / input[1]]])

    n = input.shape[0] - 1

    output = torch.zeros([n, n], dtype=input.dtype)

    scale = torch.hstack(
        [
            torch.tensor([1.0]),
            1.0 / torch.sqrt(torch.arange(n - 1, 0, -1)),
        ],
    )

    scale = torch.cumprod(scale, dim=0)
    scale = torch.flip(scale, dims=[0])

    shape = output.shape

    output = torch.reshape(output, [-1])

    output[1 :: n + 1] = torch.sqrt(torch.arange(1, n))
    output[n :: n + 1] = torch.sqrt(torch.arange(1, n))

    output = torch.reshape(output, shape)

    output[:, -1] += -scale * input[:-1] / input[-1]

    return output
