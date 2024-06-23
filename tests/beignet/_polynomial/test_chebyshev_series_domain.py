import beignet.polynomial
import torch


def test_chebyshev_series_domain():
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_series_domain,
        torch.tensor([-1.0, 1.0]),
    )
