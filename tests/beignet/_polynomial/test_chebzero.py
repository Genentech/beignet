import beignet.polynomial
import beignet.polynomial._chebyshev_series_zero
import torch


def test_chebzero():
    torch.testing.assert_close(
        beignet.polynomial._chebzero.chebyshev_series_zero,
        torch.tensor([0]),
    )
