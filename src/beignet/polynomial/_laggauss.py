import torch

from beignet.polynomial import lagcompanion, lagval
from beignet.polynomial._lagder import lagder


def laggauss(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

    c = torch.zeros(degree + 1)
    c[-1] = 1.0

    m = lagcompanion(c)
    x = torch.linalg.eigvalsh(m)

    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x = x - (dy / df)

    fm = lagval(x, c[1:])
    fm = fm / torch.abs(fm).max()
    df = df / torch.abs(df).max()
    w = 1 / (fm * df)

    w = w / torch.sum(w)

    return x, w
