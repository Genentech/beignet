import beignet.polynomial
import torch


def test_hermeline():
    torch.testing.assert_close(beignet.polynomial.hermeline(3, 4), [3, 4])
