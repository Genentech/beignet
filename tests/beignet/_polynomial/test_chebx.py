import beignet.polynomial
import beignet.polynomial._chebyshev_series_x
import torch


def test_chebx():
    torch.testing.assert_close(
        beignet.polynomial._chebx.chebyshev_series_x,
        torch.tensor([0, 1]),
    )
