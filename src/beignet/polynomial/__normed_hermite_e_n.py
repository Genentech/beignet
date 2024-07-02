import math

import torch
from torch import Tensor


def _normed_hermite_e_n(
    x: Tensor,
    n,
) -> Tensor:
    if n == 0:
        output = torch.full(x.shape, 1.0 / math.sqrt(math.sqrt(2.0 * math.pi)))
    else:
        a = torch.zeros_like(x)
        b = torch.ones_like(x) / math.sqrt(math.sqrt(2.0 * math.pi))

        size = torch.tensor(n)

        for _ in range(0, n - 1):
            previous = a

            a = -b * torch.sqrt((size - 1.0) / size)

            b = previous + b * x * torch.sqrt(1.0 / size)

            size = size - 1.0

        output = a + b * x

    return output
