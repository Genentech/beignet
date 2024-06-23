import beignet.polynomial
import torch


def test_laguerre_series_zero():
    torch.testing.assert_close(
        beignet.polynomial.laguerre_series_zero,
        torch.tensor([0]),
    )
