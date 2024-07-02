import math

import torch

from .__normed_hermite_e_n import _normed_hermite_e_n
from ._hermecompanion import hermecompanion


def hermegauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

    c = torch.zeros(degree + 1)
    c[-1] = 1.0
    m = hermecompanion(c)
    x = torch.linalg.eigvalsh(m)

    dy = _normed_hermite_e_n(x, degree)
    df = _normed_hermite_e_n(x, degree - 1) * math.sqrt(degree)
    x -= dy / df

    fm = _normed_hermite_e_n(x, degree - 1)
    fm /= torch.abs(fm).max()
    w = 1 / (fm * fm)

    a = torch.flip(w, dims=[0])
    b = torch.flip(x, dims=[0])

    w = (w + a) / 2
    x = (x - b) / 2

    w *= math.sqrt(2 * math.pi) / torch.sum(w)

    return x, w
