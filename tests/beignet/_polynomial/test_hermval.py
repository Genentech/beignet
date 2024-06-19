import beignet.polynomial
import beignet.polynomial._hermval
import beignet.polynomial._polyval
import numpy
import torch

from tests.beignet._polynomial.test_polynomial import hermite_polynomial_Hlist


def test_hermval():
    torch.testing.assert_close(beignet.polynomial._hermval.hermval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [beignet.polynomial._polyval.polyval(x, c) for c in hermite_polynomial_Hlist]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial._hermval.hermval(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial._hermval.hermval(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial._hermval.hermval(x, [1, 0]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial._hermval.hermval(x, [1, 0, 0]).shape, dims
        )
