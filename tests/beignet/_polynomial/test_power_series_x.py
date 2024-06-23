import beignet.polynomial
import torch


def test_power_series_x():
    torch.testing.assert_close(
        beignet.polynomial.power_series_x,
        torch.tensor([0, 1]),
    )
