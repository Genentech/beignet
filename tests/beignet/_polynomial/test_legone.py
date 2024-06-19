import beignet.polynomial
import torch


def test_legone():
    torch.testing.assert_close(
        beignet.polynomial.legone,
        torch.tensor([1]),
    )
