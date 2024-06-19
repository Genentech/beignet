import beignet.polynomial
import beignet.polynomial._lagval
import beignet.polynomial._polyval
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import laguerre_polynomial_Llist


def test_lagval():
    x = numpy.random.random((3, 5)) * 2 - 1

    torch.testing.assert_close(beignet.polynomial._lagval.lagval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [beignet.polynomial._polyval.polyval(x, c) for c in laguerre_polynomial_Llist]
    for i in range(7):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial._lagval.lagval(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial._lagval.lagval(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial._lagval.lagval(x, [1, 0]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial._lagval.lagval(x, [1, 0, 0]).shape, dims
        )
