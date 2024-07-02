import torch

from ._legcompanion import legcompanion
from ._legder import legder
from ._legval import legval


def leggauss(degree):
    degree = int(degree)

    if degree <= 0:
        raise ValueError

    c = torch.zeros(degree + 1)
    c[-1] = 1.0
    m = legcompanion(c)
    x = torch.linalg.eigvalsh(m)

    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df

    fm = legval(x, c[1:])

    fm /= torch.abs(fm).max()
    df /= torch.abs(df).max()

    w = 1 / (fm * df)

    a = torch.flip(w, dims=[0])
    b = torch.flip(x, dims=[0])

    w = (w + a) / 2
    x = (x - b) / 2

    w = w * (2.0 / torch.sum(w))

    return x, w
