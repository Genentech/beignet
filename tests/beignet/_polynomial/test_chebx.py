import beignet.polynomial
import beignet.polynomial._chebx
import torch


def test_chebx():
    torch.testing.assert_close(
        beignet.polynomial._chebx.chebx,
        torch.tensor([0, 1]),
    )
