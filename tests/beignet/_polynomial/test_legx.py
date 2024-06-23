import beignet.polynomial
import beignet.polynomial._legx
import torch


def test_legx():
    torch.testing.assert_close(
        beignet.polynomial._legx.legendre_series_x,
        torch.tensor([0, 1]),
    )
