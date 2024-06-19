import beignet.polynomial
import torch


def test_chebdomain():
    torch.testing.assert_close(beignet.polynomial.chebdomain, [-1, 1])
