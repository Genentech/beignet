import beignet.polynomial
import beignet.polynomial._legline
import torch


def test_legline():
    torch.testing.assert_close(
        beignet.polynomial._legline.legline(3, 4),
        torch.tensor([3, 4]),
    )

    torch.testing.assert_close(
        beignet.polynomial._legline.legline(3, 0),
        torch.tensor([3]),
    )
