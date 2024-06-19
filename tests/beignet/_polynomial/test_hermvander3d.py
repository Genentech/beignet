import beignet.polynomial
import beignet.polynomial._hermval3d
import beignet.polynomial._hermvander3d
import numpy


def test_hermvander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.polynomial._hermvander3d.hermvander3d(x1, x2, x3, [1, 2, 3])
    tgt = beignet.polynomial._hermval3d.hermval3d(x1, x2, x3, c)
    res = numpy.dot(van, c.flat)
    numpy.testing.assert_almost_equal(res, tgt)

    van = beignet.polynomial._hermvander3d.hermvander3d([x1], [x2], [x3], [1, 2, 3])
    numpy.testing.assert_(van.shape == (1, 5, 24))
