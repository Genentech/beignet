import beignet.polynomial
import beignet.polynomial._laguerre_series_x
import torch


def test_laguerre_series_x():
    torch.testing.assert_close(
        beignet.polynomial.laguerre_series_x,
        torch.tensor([1, -1]),
    )
