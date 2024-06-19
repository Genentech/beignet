import beignet.polynomial
import beignet.polynomial._hermcompanion
import torch


def test_hermcompanion():
    for i in range(1, 5):
        coef = torch.tensor([0] * i + [1])

        assert beignet.polynomial._hermcompanion.hermcompanion(coef).shape == (i, i)

    output = beignet.polynomial._hermcompanion.hermcompanion(torch.tensor([1, 2]))

    assert output[0, 0] == -0.25
