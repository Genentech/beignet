import beignet.polynomial
import beignet.polynomial._polygrid3d
import beignet.polynomial._polyval
import numpy


def test_polygrid3d():
    c1d = numpy.array([1.0, 2.0, 3.0])
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial._polyval.polyval(x, [1.0, 2.0, 3.0])

    numpy.testing.assert_almost_equal(
        beignet.polynomial._polygrid3d.polygrid3d(x1, x2, x3, c3d),
        numpy.einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(
        beignet.polynomial._polygrid3d.polygrid3d(z, z, z, c3d).shape == (2, 3) * 3
    )
