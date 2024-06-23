import operator

import numpy
import torch
from torch import Tensor

from .__normalize_axis_index import _normalize_axis_index
from ._evaluate_chebyshev_series_1d import evaluate_chebyshev_series_1d


def integrate_chebyshev_series(
    input: Tensor,
    m=1,
    k=None,
    lbnd=0,
    scl=1,
    axis=0,
):
    if k is None:
        k = []
    input = torch.ravel(input)

    # if c.dtype.char in "?bBhHiIlLqQpP":
    #     c = c.astype(numpy.double)

    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = _normalize_axis_index(iaxis, input.ndim)

    if cnt == 0:
        return input

    input = torch.moveaxis(input, iaxis, 0)

    k = list(k) + [0] * (cnt - len(k))

    for i in range(cnt):
        n = len(input)

        input *= scl

        if n == 1 and torch.all(input[0] == 0):
            input[0] += k[i]
        else:
            tmp = torch.empty((n + 1,) + input.shape[1:], dtype=input.dtype)

            tmp[0] = input[0] * 0
            tmp[1] = input[0]

            if n > 1:
                tmp[2] = input[1] / 4

            for j in range(2, n):
                tmp[j + 1] = input[j] / (2 * (j + 1))
                tmp[j - 1] -= input[j] / (2 * (j - 1))

            tmp[0] += k[i] - evaluate_chebyshev_series_1d(lbnd, tmp)

            input = tmp

    input = torch.moveaxis(input, 0, iaxis)

    return input
