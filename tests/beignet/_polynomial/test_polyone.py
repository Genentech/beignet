import beignet.polynomial
import torch


def test_polyone():
    torch.testing.assert_close(beignet.polynomial.polyone, [1])
