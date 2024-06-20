import beignet.polynomial
import numpy
import torch.testing


def test_chebgauss():
    x, w = beignet.polynomial.chebgauss(100)

    v = beignet.polynomial.chebvander(x, 99)
    vv = numpy.dot(v.T * w, v)
    vd = 1 / numpy.sqrt(vv.diagonal())
    torch.testing.assert_close(vd[:, None] * vv * vd, numpy.eye(100))

    torch.testing.assert_close(torch.sum(w), torch.pi)
