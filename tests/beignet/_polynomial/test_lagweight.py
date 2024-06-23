import beignet.polynomial
import beignet.polynomial._lagweight
import torch


def test_lagweight():
    x = torch.linspace(0, 10, 11)

    torch.testing.assert_close(
        beignet.polynomial._lagweight.laguerre_series_weight(x),
        torch.exp(-x),
    )
