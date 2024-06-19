import beignet.polynomial
import beignet.polynomial._chebval3d
import beignet.polynomial._chebvander3d
import numpy


def test_chebvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    numpy.testing.assert_almost_equal(
        numpy.dot(
            beignet.polynomial._chebvander3d.chebvander3d(x1, x2, x3, [1, 2, 3]), c.flat
        ),
        beignet.polynomial._chebval3d.chebval3d(x1, x2, x3, c),
    )
    numpy.testing.assert_(
        beignet.polynomial._chebvander3d.chebvander3d([x1], [x2], [x3], [1, 2, 3]).shape
        == (1, 5, 24)
    )
