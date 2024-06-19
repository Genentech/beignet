import beignet.polynomial
import torch


def test_lagdomain():
    torch.testing.assert_close(
        beignet.polynomial.lagdomain,
        torch.tensor([0.0, 1.0]),
    )
