import math

import beignet
import torch


def test_chebyshev_gauss_quadrature():
    output, weight = beignet.chebyshev_gauss_quadrature(100)

    output = beignet.chebyshev_polynomial_vandermonde(
        output,
        degree=torch.tensor([99]),
    )

    u = (output.T * weight) @ output

    v = 1 / torch.sqrt(u.diagonal())

    torch.testing.assert_close(
        v[:, None] * u * v,
        torch.eye(100),
    )

    torch.testing.assert_close(
        torch.sum(weight),
        torch.tensor(math.pi),
    )
