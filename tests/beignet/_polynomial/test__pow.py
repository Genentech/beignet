import beignet.polynomial
import beignet.polynomial.__pow
import numpy
import torch


def test__pow():
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.__pow._pow,
        (),
        torch.tensor([1, 2, 3]),
        5,
        4,
    )
