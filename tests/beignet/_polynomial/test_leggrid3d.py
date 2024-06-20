import beignet.polynomial
import beignet.polynomial._evaluate_1d_power_series
import beignet.polynomial._leggrid3d
import numpy


def test_leggrid3d():
    c1d = numpy.array([2.0, 2.0, 2.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial._polyval.evaluate_1d_power_series(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    res = beignet.polynomial._leggrid3d.leggrid3d(x1, x2, x3, c3d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial._leggrid3d.leggrid3d(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3) * 3)
