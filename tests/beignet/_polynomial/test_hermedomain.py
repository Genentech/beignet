import beignet.polynomial
import torch


def test_hermedomain():
    torch.testing.assert_close(beignet.polynomial.hermedomain, [-1, 1])
