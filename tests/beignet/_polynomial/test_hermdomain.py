import beignet.polynomial
import torch


def test_hermdomain():
    torch.testing.assert_close(beignet.polynomial.hermdomain, [-1, 1])
