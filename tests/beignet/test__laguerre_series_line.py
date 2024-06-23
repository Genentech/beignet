import beignet.polynomial
import beignet.polynomial._laguerre_series_line
import torch


def test_laguerre_series_line():
    torch.testing.assert_close(
        beignet.polynomial.laguerre_series_line(3, 4),
        torch.tensor([7, -4]),
    )
