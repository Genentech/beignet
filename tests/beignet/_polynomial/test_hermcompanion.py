import beignet.polynomial
import torch


def test_hermcompanion():
    for i in range(1, 5):
        coef = torch.tensor([0] * i + [1])

        assert beignet.polynomial.hermcompanion(coef).shape == (i, i)

    output = beignet.polynomial.hermcompanion(torch.tensor([1, 2]))

    assert output[0, 0] == -0.25
