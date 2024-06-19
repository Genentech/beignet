import beignet.polynomial
import numpy


def test_chebgauss():
    x, w = beignet.polynomial.chebgauss(100)

    v = beignet.polynomial.chebvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    numpy.testing.assert_almost_equal(vd[:, None] * vv * vd, numpy.eye(100))

    numpy.testing.assert_almost_equal(numpy.sum(w), numpy.pi)
