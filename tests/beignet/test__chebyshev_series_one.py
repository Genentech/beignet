import beignet.polynomial
import torch


def test_chebyshev_series_one():
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_one,
        torch.tensor([1]),
    )
