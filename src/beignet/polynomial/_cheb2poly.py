import torch
from torch import Tensor

from beignet.polynomial import _as_series, polyadd, polymulx, polysub


def cheb2poly(input: Tensor) -> Tensor:
    [input] = _as_series([input])

    n = input.shape[0]

    if n < 3:
        return input

    c0 = torch.zeros_like(input)
    c0[0] = input[-2]

    c1 = torch.zeros_like(input)
    c1[0] = input[-1]

    for index in range(0, n - 2):
        i1 = n - 1 - index

        tmp = c0

        c0 = polysub(input[i1 - 2], c1)

        c1 = polyadd(tmp, polymulx(c1, "same") * 2)

    output = polymulx(c1, "same")

    output = polyadd(c0, output)

    return output
