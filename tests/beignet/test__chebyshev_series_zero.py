import beignet.polynomial
import beignet.polynomial._chebyshev_series_zero
import torch


def test_chebyshev_series_zero():
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_zero,
        torch.tensor([0]),
    )
