import beignet.polynomial
import torch


def test_power_series_domain():
    torch.testing.assert_close(
        beignet.polynomial.power_series_domain,
        torch.tensor([-1.0, 1.0]),
    )
