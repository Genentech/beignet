import beignet.polynomial
import numpy
import torch


def test_leggauss():
    x, w = beignet.polynomial.gauss_legendre_quadrature(torch.tensor(100))

    v = beignet.polynomial.legendre_series_vandermonde_1d(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(vv, numpy.eye(100))

    tgt = 2.0
    torch.testing.assert_close(w.sum(), tgt)
