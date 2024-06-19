import beignet.polynomial
import numpy
import torch


def test_lagmulx():
    torch.testing.assert_close(beignet.polynomial.lagmulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.lagmulx([1]), [1, -1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]
        numpy.testing.assert_almost_equal(beignet.polynomial.lagmulx(ser), tgt)
