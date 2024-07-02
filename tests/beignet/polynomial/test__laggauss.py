import beignet.polynomial
import torch


def test_laggauss():
    x, w = beignet.polynomial.laggauss(100)

    v = beignet.polynomial.lagvander(x, 99)
    vv = (v.T * w) @ v
    vd = 1 / torch.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(
        vv,
        torch.eye(100),
    )

    target = 1.0
    torch.testing.assert_close(w.sum(), target)
