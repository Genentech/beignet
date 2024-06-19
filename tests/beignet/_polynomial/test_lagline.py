import beignet.polynomial
import torch


def test_lagline():
    torch.testing.assert_close(
        beignet.polynomial.lagline(3, 4),
        torch.tensor([7, -4]),
    )
