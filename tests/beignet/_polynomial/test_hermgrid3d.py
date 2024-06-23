import beignet.polynomial
import numpy
import torch


def test_hermgrid3d():
    c1d = numpy.array([2.5, 1.0, 0.75])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial._polyval.evaluate_power_series_1d(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.polynomial._hermgrid3d.hermgrid3d(x1, x2, x3, c3d)
    torch.testing.assert_close(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial._hermgrid3d.hermgrid3d(z, z, z, c3d)
    assert res.shape == (2, 3) * 3
