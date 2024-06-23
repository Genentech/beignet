import beignet.polynomial
import numpy
import torch


def test_laggauss():
    x, w = beignet.polynomial._laggauss.laggauss(100)

    v = beignet.polynomial.laguerre_series_vandermonde_1d(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(vv, numpy.eye(100))
    torch.testing.assert_close(w.sum(), 1.0)
