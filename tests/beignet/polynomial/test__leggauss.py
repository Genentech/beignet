import beignet.polynomial
import torch


def test_leggauss():
    x, w = beignet.polynomial.leggauss(100)

    v = beignet.polynomial.legvander(
        x,
        degree=torch.tensor([99]),
    )

    vv = (v.T * w) @ v

    vd = 1 / torch.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd

    torch.testing.assert_close(
        vv,
        torch.eye(100),
    )

    torch.testing.assert_close(w.sum(), 2.0)
