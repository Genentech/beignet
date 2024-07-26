import torch

from ._differentiate_legendre_polynomial import differentiate_legendre_polynomial
from ._evaluate_legendre_polynomial import evaluate_legendre_polynomial
from ._legendre_polynomial_companion import legendre_polynomial_companion


def gauss_legendre_quadrature(degree):
    degree = int(degree)

    if degree <= 0:
        raise ValueError

    c = torch.zeros(degree + 1)
    c[-1] = 1.0
    m = legendre_polynomial_companion(c)
    x = torch.linalg.eigvalsh(m)

    dy = evaluate_legendre_polynomial(x, c)
    df = evaluate_legendre_polynomial(x, differentiate_legendre_polynomial(c))
    x -= dy / df

    fm = evaluate_legendre_polynomial(x, c[1:])

    fm /= torch.abs(fm).max()
    df /= torch.abs(df).max()

    w = 1 / (fm * df)

    a = torch.flip(w, dims=[0])
    b = torch.flip(x, dims=[0])

    w = (w + a) / 2
    x = (x - b) / 2

    w = w * (2.0 / torch.sum(w))

    return x, w
