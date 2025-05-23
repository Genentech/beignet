import math

import torch

import beignet


def test_gauss_probabilists_hermite_polynomial_quadrature(float64):
    x, w = beignet.gauss_probabilists_hermite_polynomial_quadrature(100)

    v = beignet.probabilists_hermite_polynomial_vandermonde(x, 99)
    vv = (v.T * w) @ v
    vd = 1 / torch.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(vv, torch.eye(100))

    target = math.sqrt(2 * math.pi)
    torch.testing.assert_close(
        w.sum(),
        torch.tensor(target),
    )
