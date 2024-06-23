import beignet.polynomial
import torch


def test_chebcompanion():
    for index in range(1, 5):
        output = beignet.polynomial.chebyshev_series_companion(
            torch.tensor([0] * index + [1])
        )

        assert output.shape == (index, index)

    output = beignet.polynomial.chebyshev_series_companion(torch.tensor([1, 2]))

    assert output[0, 0] == -0.5
