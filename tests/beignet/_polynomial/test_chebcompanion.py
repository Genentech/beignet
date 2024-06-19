import beignet.polynomial
import beignet.polynomial._chebcompanion
import torch


def test_chebcompanion():
    for index in range(1, 5):
        output = beignet.polynomial._chebcompanion.chebcompanion(
            torch.tensor([0] * index + [1])
        )

        assert output.shape == (index, index)

    output = beignet.polynomial._chebcompanion.chebcompanion(torch.tensor([1, 2]))

    assert output[0, 0] == -0.5
