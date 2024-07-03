import math

import torch

from ._probabilists_hermite_polynomial_companion import (
    probabilists_hermite_polynomial_companion,
)


def gauss_probabilists_hermite_polynomial_quadrature(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

    c = torch.zeros(degree + 1)
    c[-1] = 1.0
    m = probabilists_hermite_polynomial_companion(c)
    x = torch.linalg.eigvalsh(m)

    if degree == 0:
        dy = torch.full(x.shape, 1.0 / math.sqrt(math.sqrt(2.0 * math.pi)))
    else:
        a1 = torch.zeros_like(x)
        b1 = torch.ones_like(x) / math.sqrt(math.sqrt(2.0 * math.pi))

        size = torch.tensor(degree)

        for _ in range(0, degree - 1):
            previous = a1

            a1 = -b1 * torch.sqrt((size - 1.0) / size)

            b1 = previous + b1 * x * torch.sqrt(1.0 / size)

            size = size - 1.0

        dy = a1 + b1 * x

    n = degree - 1

    if n == 0:
        df = torch.full(x.shape, 1 / math.sqrt(math.sqrt(math.pi)))
    else:
        a = torch.zeros_like(x)

        b = torch.ones_like(x) / math.sqrt(math.sqrt(math.pi))

        size = torch.tensor(n)

        for _ in range(0, n - 1):
            previous = a

            a = -b * torch.sqrt((size - 1.0) / size)

            b = previous + b * x * torch.sqrt(2.0 / size)

            size = size - 1.0

        df = a + b * x * math.sqrt(2.0)

    df = df * math.sqrt(degree)

    x = x - (dy / df)

    n = degree - 1

    if n == 0:
        fm = torch.full(x.shape, 1.0 / math.sqrt(math.sqrt(2.0 * math.pi)))
    else:
        a = torch.zeros_like(x)
        b = torch.ones_like(x) / math.sqrt(math.sqrt(2.0 * math.pi))

        size = torch.tensor(n)

        for _ in range(0, n - 1):
            previous = a

            a = -b * torch.sqrt((size - 1.0) / size)

            b = previous + b * x * torch.sqrt(1.0 / size)

            size = size - 1.0

        fm = a + b * x

    fm = fm / torch.abs(fm).max()

    w = 1 / (fm * fm)

    a = torch.flip(w, dims=[0])
    b = torch.flip(x, dims=[0])

    w = (w + a) / 2
    x = (x - b) / 2

    w = w * (math.sqrt(2 * math.pi) / torch.sum(w))

    return x, w
