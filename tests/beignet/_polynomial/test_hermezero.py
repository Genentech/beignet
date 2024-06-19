import beignet.polynomial
import torch


def test_hermezero():
    torch.testing.assert_close(beignet.polynomial.hermezero, [0])
