import math

import beignet.polynomial
import torch


def test_hermegauss():
    x, w = beignet.polynomial.hermegauss(100)

    v = beignet.polynomial.hermevander(x, 99)
    vv = (v.T * w) @ v
    vd = 1 / torch.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(vv, torch.eye(100))

    target = math.sqrt(2 * math.pi)
    torch.testing.assert_close(
        w.sum(),
        torch.tensor(target),
    )
