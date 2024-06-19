import beignet.polynomial
import beignet.polynomial._laggauss
import beignet.polynomial._lagvander
import numpy


def test_laggauss():
    x, w = beignet.polynomial._laggauss.laggauss(100)

    v = beignet.polynomial._lagvander.lagvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_almost_equal(vv, numpy.eye(100))
    numpy.testing.assert_almost_equal(w.sum(), 1.0)
