import beignet.polynomial
import beignet.polynomial._legval3d
import beignet.polynomial._polyval
import numpy


def test_legval3d():
    c1d = numpy.array([2.0, 2.0, 2.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial._polyval.polyval(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legval3d.legval3d, x1, x2, x3[:2], c3d
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial._legval3d.legval3d(x1, x2, x3, c3d), y1 * y2 * y3
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(
        beignet.polynomial._legval3d.legval3d(z, z, z, c3d).shape == (2, 3)
    )
