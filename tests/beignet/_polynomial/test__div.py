import beignet.polynomial
import beignet.polynomial.__div
import numpy
import torch


def test__div():
    numpy.testing.assert_raises(
        ZeroDivisionError,
        beignet.polynomial.__div._div,
        beignet.polynomial.__div._div,
        torch.tensor([1, 2, 3]),
        torch.tensor([0]),
    )
