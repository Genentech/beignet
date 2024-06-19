import beignet.polynomial
import torch


def test_chebzero():
    torch.testing.assert_close(beignet.polynomial.chebzero, [0])
