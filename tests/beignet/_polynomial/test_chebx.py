import beignet.polynomial
import torch


def test_chebx():
    torch.testing.assert_close(
        beignet.polynomial.chebx,
        torch.tensor([0, 1]),
    )
