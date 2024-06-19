import beignet.polynomial
import numpy
import torch


def test_legtrim():
    coef = torch.tensor([2, -1, 1, 0], dtype=torch.float64)

    numpy.testing.assert_raises(ValueError, beignet.polynomial.legtrim, coef, -1)

    torch.testing.assert_close(
        beignet.polynomial.legtrim(coef),
        coef[:-1],
    )

    torch.testing.assert_close(
        beignet.polynomial.legtrim(coef, 1),
        coef[:-3],
    )

    torch.testing.assert_close(
        beignet.polynomial.legtrim(coef, 2),
        torch.tensor([0], dtype=torch.float64),
    )
