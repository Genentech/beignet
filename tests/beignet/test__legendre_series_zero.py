import beignet.polynomial
import beignet.polynomial._legendre_series_zero
import torch


def test_legendre_series_zero():
    torch.testing.assert_close(
        beignet.polynomial.legendre_series_zero,
        torch.tensor([0]),
    )
