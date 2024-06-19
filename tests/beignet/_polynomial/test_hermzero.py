import beignet.polynomial
import torch


def test_hermzero():
    torch.testing.assert_close(beignet.polynomial.hermzero, [0])
