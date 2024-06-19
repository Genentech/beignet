import beignet._polynomial.__pow
import beignet.polynomial
import numpy
import torch


def test__pow():
    numpy.testing.assert_raises(
        ValueError,
        beignet._polynomial.__pow._pow,
        (),
        torch.tensor([1, 2, 3]),
        5,
        4,
    )
