import beignet.polynomial
import torch.testing


def test_power_series_zero():
    torch.testing.assert_close(
        beignet.polynomial.power_series_zero,
        torch.tensor([0]),
    )
