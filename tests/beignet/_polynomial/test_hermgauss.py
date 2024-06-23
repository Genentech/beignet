import beignet.polynomial
import numpy


def test_hermgauss():
    x, w = beignet.polynomial.hermgauss(100)

    v = beignet.polynomial.physicists_hermite_series_vandermonde_1d(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    numpy.testing.assert_almost_equal(vv, numpy.eye(100))

    tgt = numpy.sqrt(numpy.pi)
    numpy.testing.assert_almost_equal(w.sum(), tgt)
