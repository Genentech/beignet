import beignet.polynomial
import numpy


def test_polyvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random([2, 3, 4])

    numpy.testing.assert_almost_equal(
        numpy.dot(beignet.polynomial.polyvander3d(x1, x2, x3, [1, 2, 3]), c.flat),
        beignet.polynomial.polyval3d(x1, x2, x3, c),
    )

    output = beignet.polynomial.polyvander3d([x1], [x2], [x3], [1, 2, 3])

    assert output.shape == (1, 5, 24)
