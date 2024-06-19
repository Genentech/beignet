import beignet.polynomial
import torch


def test_hermex():
    torch.testing.assert_close(beignet.polynomial.hermex, [0, 1])
