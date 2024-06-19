import beignet.polynomial
import beignet.polynomial._hermex
import torch


def test_hermex():
    torch.testing.assert_close(beignet.polynomial._hermex.hermex, [0, 1])
