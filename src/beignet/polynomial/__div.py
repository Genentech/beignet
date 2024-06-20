import torch

from .__as_series import _as_series
from .__trim_sequence import _trim_sequence


def _div(func, a, b):
    a, b = _as_series([a, b])

    if b[-1] == 0:
        raise ZeroDivisionError

    m = a.shape[-1]
    n = b.shape[-1]

    if m < n:
        quotient, remainder = a[:1] * 0.0, a
    elif n == 1:
        quotient, remainder = a / b[-1], a[:1] * 0.0
    else:
        quotient = torch.empty(m - n + 1, dtype=a.dtype)

        remainder = a

        for index in range(m - n, -1, -1):
            shape = [0] * index

            p = func(torch.tensor([*shape, 1]), b)

            q = remainder[-1] / p[-1]

            remainder = remainder[:-1] - q * p[:-1]

            quotient[index] = q

        remainder = _trim_sequence(remainder)

    return quotient, remainder
