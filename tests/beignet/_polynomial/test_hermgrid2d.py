import beignet.polynomial
import beignet.polynomial._hermgrid2d
import beignet.polynomial._polyval
import numpy


def test_hermgrid2d():
    c1d = numpy.array([2.5, 1.0, 0.75])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial._polyval.polyval(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j->ij", y1, y2)
    res = beignet.polynomial._hermgrid2d.hermgrid2d(x1, x2, c2d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial._hermgrid2d.hermgrid2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3) * 2)
