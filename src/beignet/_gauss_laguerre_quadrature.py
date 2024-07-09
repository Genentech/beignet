import torch

from ._differentiate_laguerre_polynomial import differentiate_laguerre_polynomial
from ._evaluate_laguerre_polynomial import evaluate_laguerre_polynomial
from ._laguerre_polynomial_companion import laguerre_polynomial_companion


def gauss_laguerre_quadrature(degree):
    degree = int(degree)
    if degree <= 0:
        raise ValueError

    c = torch.zeros(degree + 1)
    c[-1] = 1.0

    m = laguerre_polynomial_companion(c)
    x = torch.linalg.eigvalsh(m)

    dy = evaluate_laguerre_polynomial(x, c)
    df = evaluate_laguerre_polynomial(x, differentiate_laguerre_polynomial(c))
    x = x - (dy / df)

    fm = evaluate_laguerre_polynomial(x, c[1:])
    fm = fm / torch.abs(fm).max()
    df = df / torch.abs(df).max()
    w = 1 / (fm * df)

    w = w / torch.sum(w)

    return x, w
