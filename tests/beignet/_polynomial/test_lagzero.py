import beignet.polynomial
import torch


def test_lagzero():
    torch.testing.assert_close(beignet.polynomial.lagzero, [0])
