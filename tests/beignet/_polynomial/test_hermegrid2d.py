import beignet.polynomial
import numpy
import torch


def test_hermegrid2d():
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial._polyval.evaluate_power_series_1d(
        x, [1.0, 2.0, 3.0]
    )

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.polynomial._hermegrid2d.hermegrid2d(x1, x2, c2d)
    torch.testing.assert_close(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial._hermegrid2d.hermegrid2d(z, z, c2d)
    assert res.shape == (2, 3) * 2
