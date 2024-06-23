import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_companion
import torch


def test_hermcompanion():
    for i in range(1, 5):
        coef = torch.tensor([0] * i + [1])

        assert beignet.polynomial._hermcompanion.physicists_hermite_series_companion(
            coef
        ).shape == (i, i)

    output = beignet.polynomial._hermcompanion.physicists_hermite_series_companion(
        torch.tensor([1, 2])
    )

    assert output[0, 0] == -0.25
