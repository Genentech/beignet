import beignet.polynomial
import beignet.polynomial._chebyshev_series_one
import torch


def test_chebone():
    torch.testing.assert_close(
        beignet.polynomial._chebone.chebyshev_series_one,
        torch.tensor([1]),
    )
