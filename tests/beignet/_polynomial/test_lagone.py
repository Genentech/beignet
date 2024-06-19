import beignet.polynomial
import torch


def test_lagone():
    torch.testing.assert_close(
        beignet.polynomial.lagone,
        torch.tensor([1]),
    )
