import beignet.polynomial
import numpy
import torch


def test_hermevander3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    van = beignet.polynomial._hermevander3d.probabilists_hermite_series_vandermonde_3d(
        x1, x2, x3, [1, 2, 3]
    )
    tgt = beignet.polynomial._hermeval3d.evaluate_probabilists_hermite_series_3d(
        x1, x2, x3, c
    )
    res = numpy.dot(van, c.flat)
    torch.testing.assert_close(res, tgt)

    van = beignet.polynomial._hermevander3d.probabilists_hermite_series_vandermonde_3d(
        [x1], [x2], [x3], [1, 2, 3]
    )
    assert van.shape == (1, 5, 24)
