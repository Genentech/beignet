import beignet.polynomial
import beignet.polynomial._polyval
import numpy
import torch


def test_polyval():
    torch.testing.assert_close(beignet.polynomial._polyval.polyval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [x**i for i in range(5)]
    for i in range(5):
        tgt = y[i]
        numpy.testing.assert_almost_equal(
            beignet.polynomial._polyval.polyval(x, [0] * i + [1]), tgt
        )
    tgt = x * (x**2 - 1)
    numpy.testing.assert_almost_equal(
        beignet.polynomial._polyval.polyval(x, [0, -1, 0, 1]), tgt
    )

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial._polyval.polyval(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial._polyval.polyval(x, [1, 0]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial._polyval.polyval(x, [1, 0, 0]).shape, dims
        )
