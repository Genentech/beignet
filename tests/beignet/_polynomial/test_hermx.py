import beignet.polynomial
import torch


def test_hermx():
    torch.testing.assert_close(beignet.polynomial.hermx, [0, 0.5])
