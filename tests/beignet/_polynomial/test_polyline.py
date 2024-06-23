import beignet.polynomial
import torch


def test_polyline():
    torch.testing.assert_close(
        beignet.polynomial.power_series_line(3, 4),
        torch.tensor([3, 4]),
    )

    torch.testing.assert_close(
        beignet.polynomial.power_series_line(3, 0),
        torch.tensor([3]),
    )
