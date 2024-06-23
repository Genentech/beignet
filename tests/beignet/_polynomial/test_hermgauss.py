import beignet.polynomial
import numpy
import torch


def test_hermgauss():
    x, w = beignet.polynomial.hermgauss(100)

    v = beignet.polynomial.physicists_hermite_series_vandermonde_1d(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(vv, numpy.eye(100))

    tgt = numpy.sqrt(numpy.pi)
    torch.testing.assert_close(w.sum(), tgt)
