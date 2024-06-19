import beignet.polynomial
import torch


def test_lagdomain():
    torch.testing.assert_close(beignet.polynomial.lagdomain, [0, 1])
