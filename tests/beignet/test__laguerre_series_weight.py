import beignet.polynomial
import torch


def test_laguerre_series_weight():
    x = torch.linspace(0, 10, 11)

    torch.testing.assert_close(
        beignet.polynomial.laguerre_series_weight(x),
        torch.exp(-x),
    )
