import math

import beignet.polynomial
import torch


def test_hermgauss():
    x, w = beignet.polynomial.hermgauss(100)

    v = beignet.polynomial.hermvander(x, 99)
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
