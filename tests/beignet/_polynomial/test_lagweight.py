import beignet.polynomial
import numpy
import torch


def test_lagweight():
    x = torch.linspace(0, 10, 11)

    numpy.testing.assert_almost_equal(
        beignet.polynomial.lagweight(x),
        torch.exp(-x),
    )
