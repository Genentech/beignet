import beignet.polynomial
import beignet.polynomial._lagline
import torch


def test_lagline():
    torch.testing.assert_close(
        beignet.polynomial._lagline.lagline(3, 4),
        torch.tensor([7, -4]),
    )
