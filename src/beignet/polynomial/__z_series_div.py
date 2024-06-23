import torch
from torch import Tensor


def _z_series_div(input: Tensor, other: Tensor) -> (Tensor, Tensor):
    lc1 = len(input)
    lc2 = len(other)

    if lc2 == 1:
        input /= other

        return input, input[:1] * 0

    if lc1 < lc2:
        return input[:1] * 0, input

    dlen = lc1 - lc2

    scl = other[0]

    other = other / scl

    quotient = torch.empty(dlen + 1, dtype=input.dtype)

    j = 0

    k = dlen

    while j < k:
        r = input[j]

        quotient[j] = input[j]
        quotient[dlen - j] = r

        tmp = r * other

        input[j : j + lc2] -= tmp
        input[k : k + lc2] -= tmp

        j = j + 1
        k = k - 1

    r = input[j]

    quotient[j] = r

    tmp = r * other

    input[j : j + lc2] -= tmp

    quotient /= scl

    remainder = input[j + 1 : j - 1 + lc2]

    return quotient, remainder
