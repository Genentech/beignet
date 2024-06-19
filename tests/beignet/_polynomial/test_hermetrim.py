import beignet.polynomial
import beignet.polynomial._hermetrim
import numpy
import torch


def test_hermetrim():
    coef = [2, -1, 1, 0]

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._hermetrim.hermetrim, coef, -1
    )

    torch.testing.assert_close(beignet.polynomial._hermetrim.hermetrim(coef), coef[:-1])
    torch.testing.assert_close(
        beignet.polynomial._hermetrim.hermetrim(coef, 1), coef[:-3]
    )
    torch.testing.assert_close(beignet.polynomial._hermetrim.hermetrim(coef, 2), [0])
