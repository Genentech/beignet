import beignet.polynomial
import beignet.polynomial._chebline
import torch


def test_chebline():
    torch.testing.assert_close(
        beignet.polynomial._chebline.chebline(3, 4),
        torch.tensor([3, 4]),
    )
