import beignet.polynomial
import beignet.polynomial._legendre_series_vandermonde_1d
import beignet.polynomial._leggauss
import numpy


def test_leggauss():
    x, w = beignet.polynomial._leggauss.leggauss(100)

    v = beignet.polynomial.legendre_series_vandermonde_1d(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_almost_equal(vv, numpy.eye(100))

    tgt = 2.0
    numpy.testing.assert_almost_equal(w.sum(), tgt)
