import beignet.polynomial
import beignet.polynomial._polyline
import torch


def test_polyline():
    torch.testing.assert_close(
        beignet.polynomial._polyline.polyline(3, 4),
        torch.tensor([3, 4]),
    )

    torch.testing.assert_close(
        beignet.polynomial._polyline.polyline(3, 0),
        torch.tensor([3]),
    )
