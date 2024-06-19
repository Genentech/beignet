import beignet.polynomial
import beignet.polynomial._chebgauss
import beignet.polynomial._chebvander
import numpy


def test_chebgauss():
    x, w = beignet.polynomial._chebgauss.chebgauss(100)

    v = beignet.polynomial._chebvander.chebvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    numpy.testing.assert_almost_equal(vd[:, None] * vv * vd, numpy.eye(100))

    numpy.testing.assert_almost_equal(numpy.sum(w), numpy.pi)
