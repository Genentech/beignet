import beignet.polynomial
import torch


def test_chebyshev_series_x():
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_x,
        torch.tensor([0, 1]),
    )
