import beignet.polynomial
import torch


def test_legx():
    torch.testing.assert_close(
        beignet.polynomial.legx,
        torch.tensor([0, 1]),
    )
