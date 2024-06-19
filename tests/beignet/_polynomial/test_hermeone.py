import beignet.polynomial
import torch


def test_hermeone():
    torch.testing.assert_close(beignet.polynomial.hermeone, [1])
