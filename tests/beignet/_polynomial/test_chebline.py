import beignet.polynomial
import torch


def test_chebline():
    torch.testing.assert_close(
        beignet.polynomial.chebline(3, 4),
        torch.tensor([3, 4]),
    )
