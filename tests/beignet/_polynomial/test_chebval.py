import beignet.polynomial
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import chebyshev_polynomial_Tlist


def test_chebval():
    torch.testing.assert_close(beignet.polynomial.chebval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [beignet.polynomial.polyval(x, c) for c in chebyshev_polynomial_Tlist]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.chebval(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(beignet.polynomial.chebval(x, [1]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.chebval(x, [1, 0]).shape, dims)
        torch.testing.assert_close(beignet.polynomial.chebval(x, [1, 0, 0]).shape, dims)
