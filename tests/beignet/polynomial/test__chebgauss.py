import math

import beignet.polynomial
import torch


def test_chebgauss():
    output, weight = beignet.polynomial.chebgauss(100)

    vandermonde = beignet.polynomial.chebvander(
        output,
        degree=torch.tensor([99]),
    )

    u = (vandermonde.T * weight) @ vandermonde

    v = 1 / torch.sqrt(u.diagonal())

    torch.testing.assert_close(
        v[:, None] * u * v,
        torch.eye(100),
    )

    torch.testing.assert_close(
        weight.sum(),
        torch.tensor(math.pi),
    )
