import math
from typing import Callable

import torch
from torch import Tensor

from .__as_series import _as_series


def _from_roots(
    f: Callable,
    g: Callable,
    input: Tensor,
) -> Tensor:
    if math.prod(input.shape) == 0:
        return torch.ones([1])

    input, _ = torch.sort(input)

    ys = []

    for x in input:
        a = torch.zeros(input.shape[0] + 1, dtype=x.dtype)
        b = f(-x, 1)

        [a, b] = _as_series([a, b])

        if a.shape[0] > b.shape[0]:
            y = torch.concatenate(
                [
                    b,
                    torch.zeros(
                        a.shape[0] - b.shape[0],
                        dtype=b.dtype,
                    ),
                ],
            )

            y = a + y
        else:
            y = torch.concatenate(
                [
                    a,
                    torch.zeros(
                        b.shape[0] - a.shape[0],
                        dtype=a.dtype,
                    ),
                ]
            )

            y = b + y

        ys = [*ys, y]

    p = torch.stack(ys)

    m = p.shape[0]

    x = m, p

    while x[0] > 1:
        m, r = divmod(x[0], 2)

        z = x[1]

        previous = torch.zeros([len(p), input.shape[0] + 1])

        y = previous

        for i in range(0, m):
            y[i] = g(z[i], z[i + m])[: input.shape[0] + 1]

        previous = y

        if r:
            previous[0] = g(previous[0], z[2 * m])[: input.shape[0] + 1]

        x = m, previous

    _, output = x

    return output[0]
