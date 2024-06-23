import beignet.polynomial
import torch


def test_legendre_series_x():
    torch.testing.assert_close(
        beignet.polynomial.legendre_series_x,
        torch.tensor([0, 1]),
    )
