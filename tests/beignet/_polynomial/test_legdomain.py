import beignet.polynomial
import torch


def test_legdomain():
    torch.testing.assert_close(beignet.polynomial.legdomain, [-1, 1])
