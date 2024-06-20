import beignet.polynomial
import beignet.polynomial._evaluate_1d_power_series
import beignet.polynomial._evaluate_3d_laguerre_series
import numpy


def test_evaluate_3d_laguerre_series():
    c1d = numpy.array([9.0, -14.0, 6.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial._polyval.evaluate_1d_power_series(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._lagval3d.evaluate_3d_laguerre_series,
        x1,
        x2,
        x3[:2],
        c3d,
    )

    tgt = y1 * y2 * y3
    res = beignet.polynomial._lagval3d.evaluate_3d_laguerre_series(x1, x2, x3, c3d)
    numpy.testing.assert_almost_equal(res, tgt)

    z = numpy.ones((2, 3))
    res = beignet.polynomial._lagval3d.evaluate_3d_laguerre_series(z, z, z, c3d)
    numpy.testing.assert_(res.shape == (2, 3))
