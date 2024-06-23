import beignet.polynomial
import beignet.polynomial._lagx
import torch


def test_lagx():
    torch.testing.assert_close(
        beignet.polynomial._lagx.laguerre_series_x,
        torch.tensor([1, -1]),
    )
