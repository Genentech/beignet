import beignet.polynomial
import numpy


def test_legval2d():
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.polyval(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.legval2d, x1, x2[:2], c2d
    )

    tgt = y1 * y2
    numpy.testing.assert_almost_equal(beignet.polynomial.legval2d(x1, x2, c2d), tgt)

    z = numpy.ones((2, 3))
    numpy.testing.assert_(beignet.polynomial.legval2d(z, z, c2d).shape == (2, 3))
