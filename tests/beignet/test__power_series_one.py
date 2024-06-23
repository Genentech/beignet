import beignet.polynomial
import torch


def test_power_series_one():
    torch.testing.assert_close(
        beignet.polynomial.power_series_one,
        torch.tensor([1]),
    )
