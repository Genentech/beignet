from typing import Callable, Tuple

import torch
from torch import Tensor

from .__as_series import _as_series
from .__nonzero import _nonzero


def _div(
    func: Callable,
    input: Tensor,
    other: Tensor,
) -> Tuple[Tensor, Tensor]:
    [input, other] = _as_series([input, other])

    m = input.shape[0]
    n = other.shape[0]

    if m < n:
        return torch.zeros_like(input[:1]), input

    if n == 1:
        return input / other[-1], torch.zeros_like(input[:1])

    def f(x: Tensor) -> Tensor:
        indicies = torch.flip(x, [0])

        indicies = _nonzero(indicies, size=1)

        return x.shape[0] - 1 - indicies[0][0]

    quotient = torch.zeros(m - n + 1, dtype=input.dtype)

    ridx = input.shape[0] - 1

    size = m - f(other) - 1

    y = torch.zeros(m + n + 1, dtype=input.dtype)

    y[size] = 1.0

    x = quotient, input, y, ridx

    for index in range(0, size):
        quotient, remainder, y2, ridx1 = x

        j = size - index

        p = func(y2, other)

        pidx = f(p)

        t = remainder[ridx1] / p[pidx]

        remainder_modified = remainder.clone()
        remainder_modified[ridx1] = 0.0

        a = remainder_modified

        p_modified = p.clone()
        p_modified[pidx] = 0.0

        b = t * p_modified

        [a, b] = _as_series([a, b])

        if a.shape[0] > b.shape[0]:
            output = -b

            output = torch.concatenate(
                [
                    output,
                    torch.zeros(
                        a.shape[0] - b.shape[0],
                        dtype=b.dtype,
                    ),
                ],
            )
            output = a + output
        else:
            output = -b

            output = torch.concatenate(
                [
                    output[: a.shape[0]] + a,
                    output[a.shape[0] :],
                ],
            )

        remainder = output

        remainder = remainder[: remainder.shape[0]]

        quotient[j] = t

        ridx1 = ridx1 - 1

        y2 = torch.roll(y2, -1)

        x = quotient, remainder, y2, ridx1

    quotient, remainder, _, _ = x

    return quotient, remainder
