import beignet.polynomial
import torch


def test_physicists_hermite_series_companion():
    for index in range(1, 5):
        coef = torch.tensor([0] * index + [1])

        output = beignet.polynomial.physicists_hermite_series_companion(coef)

        assert output.shape == (index, index)

    output = beignet.polynomial.physicists_hermite_series_companion(
        torch.tensor([1, 2])
    )

    assert output[0, 0] == -0.25
