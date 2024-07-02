import math

import torch
from beignet.polynomial import chebgauss, chebvander


def test_chebgauss():
    output, weight = chebgauss(100)

    vandermonde = chebvander(
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
