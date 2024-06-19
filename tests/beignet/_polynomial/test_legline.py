import beignet.polynomial
import torch


def test_legline():
    torch.testing.assert_close(
        beignet.polynomial.legline(3, 4),
        torch.tensor([3, 4]),
    )

    torch.testing.assert_close(
        beignet.polynomial.legline(3, 0),
        torch.tensor([3]),
    )
