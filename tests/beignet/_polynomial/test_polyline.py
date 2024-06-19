import beignet.polynomial
import torch


def test_polyline():
    torch.testing.assert_close(
        beignet.polynomial.polyline(3, 4),
        torch.tensor([3, 4]),
    )

    torch.testing.assert_close(
        beignet.polynomial.polyline(3, 0),
        torch.tensor([3]),
    )
