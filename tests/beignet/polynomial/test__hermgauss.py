import math

import torch
from beignet.polynomial import hermgauss, hermvander


def test_hermgauss():
    x, w = hermgauss(100)

    v = hermvander(x, 99)
    vv = (v.T * w) @ v
    vd = 1 / torch.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(
        vv,
        torch.eye(100),
    )

    torch.testing.assert_close(
        w.sum(),
        torch.tensor(math.sqrt(math.pi)),
    )
