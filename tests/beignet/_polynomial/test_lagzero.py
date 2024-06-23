import beignet.polynomial
import beignet.polynomial._lagzero
import torch


def test_lagzero():
    torch.testing.assert_close(
        beignet.polynomial._lagzero.laguerre_series_zero,
        torch.tensor([0]),
    )
