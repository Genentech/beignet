import torch
from beignet.polynomial import polyline


def test_polyline():
    torch.testing.assert_close(
        polyline(3.0, 4.0),
        torch.tensor([3.0, 4.0]),
    )

    torch.testing.assert_close(
        polyline(3.0, 0.0),
        torch.tensor([3.0, 0.0]),
    )
