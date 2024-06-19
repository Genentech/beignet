import beignet.polynomial
import torch


def test_lagx():
    torch.testing.assert_close(beignet.polynomial.lagx, [1, -1])
