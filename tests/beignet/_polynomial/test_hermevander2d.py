import beignet.polynomial
import beignet.polynomial._hermeval2d
import beignet.polynomial._hermevander2d
import numpy


def test_hermevander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.polynomial._hermevander2d.hermevander2d(x1, x2, [1, 2])
    tgt = beignet.polynomial._hermeval2d.hermeval2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial._hermevander2d.hermevander2d([x1], [x2], [1, 2])
    numpy.testing.assert_(van.shape == (1, 5, 6))
