import beignet.polynomial
import beignet.polynomial._chebyshev_series_line
import torch


def test_chebline():
    torch.testing.assert_close(
        beignet.polynomial._chebline.chebyshev_series_line(3, 4),
        torch.tensor([3, 4]),
    )
