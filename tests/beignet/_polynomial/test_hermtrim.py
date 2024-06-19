import beignet.polynomial
import numpy
import torch


def test_hermtrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermtrim, coef, -1)

    torch.testing.assert_close(beignet.polynomial.hermtrim(coef), coef[:-1])
    torch.testing.assert_close(beignet.polynomial.hermtrim(coef, 1), coef[:-3])
    torch.testing.assert_close(beignet.polynomial.hermtrim(coef, 2), [0])
