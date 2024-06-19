import beignet.polynomial
import numpy
import torch


def test_lagtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.polynomial.lagtrim, coef, -1)

    torch.testing.assert_close(beignet.polynomial.lagtrim(coef), coef[:-1])
    torch.testing.assert_close(beignet.polynomial.lagtrim(coef, 1), coef[:-3])
    torch.testing.assert_close(beignet.polynomial.lagtrim(coef, 2), [0])
