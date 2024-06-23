import beignet.polynomial
import torch


def test_legendre_series_one():
    torch.testing.assert_close(
        beignet.polynomial.legendre_series_one,
        torch.tensor([1]),
    )
