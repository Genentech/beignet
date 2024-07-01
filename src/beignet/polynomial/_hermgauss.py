import math

import torch

from beignet.polynomial import hermcompanion
from beignet.polynomial.__normed_hermite_n import _normed_hermite_n


def hermgauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

    c = torch.zeros(degree + 1)
    c[-1] = 1.0

    x = torch.linalg.eigvalsh(hermcompanion(c))

    dy = _normed_hermite_n(x, degree)
    df = _normed_hermite_n(x, degree - 1) * math.sqrt(2 * degree)

    x = x - (dy / df)

    fm = _normed_hermite_n(x, degree - 1)
    fm = fm / torch.abs(fm).max()
    w = 1 / (fm * fm)

    a = torch.flip(w, dims=[0])
    b = torch.flip(x, dims=[0])

    w = (w + a) / 2
    x = (x - b) / 2

    w = w * (math.sqrt(math.pi) / torch.sum(w))

    return x, w
