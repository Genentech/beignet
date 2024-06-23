import beignet.polynomial
import beignet.polynomial._laguerre_series_one
import torch


def test_laguerre_series_one():
    torch.testing.assert_close(
        beignet.polynomial.laguerre_series_one,
        torch.tensor([1]),
    )
