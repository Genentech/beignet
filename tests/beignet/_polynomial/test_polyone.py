import beignet.polynomial
import torch


def test_polyone():
    torch.testing.assert_close(
        beignet.polynomial._polyone.power_series_one,
        torch.tensor([1]),
    )
