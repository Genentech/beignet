import beignet.polynomial
import torch.testing


def test_polyzero():
    torch.testing.assert_close(
        beignet.polynomial.power_series_zero,
        torch.tensor([0]),
    )
