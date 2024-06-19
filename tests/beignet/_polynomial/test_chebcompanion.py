import beignet.polynomial
import torch


def test_chebcompanion():
    for i in range(1, 5):
        assert beignet.polynomial.chebcompanion(torch.tensor([0] * i + [1])).shape == (
            i,
            i,
        )

    assert beignet.polynomial.chebcompanion([1, 2])[0, 0] == -0.5
