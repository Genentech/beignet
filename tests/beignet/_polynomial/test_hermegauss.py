import beignet.polynomial
import beignet.polynomial._hermegauss
import beignet.polynomial._probabilists_hermite_series_vandermonde_1d
import numpy
import torch


def test_hermegauss():
    x, w = beignet.polynomial._hermegauss.hermegauss(100)

    v = beignet.polynomial._hermevander.probabilists_hermite_series_vandermonde_1d(
        x, 99
    )
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    vv = vd[:, None] * vv * vd
    torch.testing.assert_close(vv, numpy.eye(100))

    tgt = numpy.sqrt(2 * numpy.pi)
    torch.testing.assert_close(w.sum(), tgt)
