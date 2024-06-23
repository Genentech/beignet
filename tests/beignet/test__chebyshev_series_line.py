import beignet.polynomial
import torch


def test_chebyshev_series_line():
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_line(3, 4),
        torch.tensor([3, 4]),
    )
