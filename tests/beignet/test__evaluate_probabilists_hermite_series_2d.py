import beignet.polynomial
import numpy


def test_evaluate_probabilists_hermite_series_2d():
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.evaluate_power_series_1d(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.evaluate_probabilists_hermite_series_2d,
        x1,
        x2[:2],
        c2d,
    )

    tgt = y1 * y2
    res = beignet.polynomial.evaluate_probabilists_hermite_series_2d(x1, x2, c2d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial.evaluate_probabilists_hermite_series_2d(z, z, c2d)
    numpy.testing.assert_(res.shape == (2, 3))
