import beignet.polynomial
import beignet.polynomial._laguerre_series_one
import torch


def test_lagone():
    torch.testing.assert_close(
        beignet.polynomial._lagone.laguerre_series_one,
        torch.tensor([1]),
    )
