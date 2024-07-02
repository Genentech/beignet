import torch
from beignet.polynomial import leggauss, legvander


def test_leggauss():
    x, w = leggauss(100)

    v = legvander(
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
