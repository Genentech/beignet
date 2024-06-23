import beignet.polynomial
import torch


def test_chebyshev_series_weight():
    x = torch.linspace(-1, 1, 11)[1:-1]
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_weight(x),
        1.0 / (torch.sqrt(1 + x) * torch.sqrt(1 - x)),
    )
