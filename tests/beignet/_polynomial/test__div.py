import beignet._polynomial.__div
import beignet.polynomial
import numpy
import torch


def test__div():
    numpy.testing.assert_raises(
        ZeroDivisionError,
        beignet._polynomial.__div._div,
        beignet._polynomial.__div._div,
        torch.tensor([1, 2, 3]),
        torch.tensor([0]),
    )
