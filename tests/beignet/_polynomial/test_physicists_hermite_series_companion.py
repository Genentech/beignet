import beignet.polynomial
import torch


def test_physicists_hermite_series_companion():
    for i in range(1, 5):
        coef = torch.tensor([0] * i + [1])

        assert beignet.polynomial.physicists_hermite_series_companion(coef).shape == (
            i,
            i,
        )

    output = beignet.polynomial.physicists_hermite_series_companion(
        torch.tensor([1, 2])
    )

    assert output[0, 0] == -0.25
