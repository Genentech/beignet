import beignet.polynomial
import beignet.polynomial._evaluate_2d_physicists_hermite_series
import beignet.polynomial._hermvander2d
import numpy


def test_hermvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.polynomial._hermvander2d.physicists_hermite_series_hermvander2d(
        x1, x2, [1, 2]
    )
    tgt = beignet.polynomial._hermval2d.evaluate_2d_physicists_hermite_series(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial._hermvander2d.physicists_hermite_series_hermvander2d(
        [x1], [x2], [1, 2]
    )
    numpy.testing.assert_(van.shape == (1, 5, 6))
