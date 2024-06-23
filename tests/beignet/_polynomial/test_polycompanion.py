import beignet.polynomial
import pytest
import torch


def test_polycompanion():
    with pytest.raises(ValueError):
        beignet.polynomial.power_series_companion(
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        beignet.polynomial.power_series_companion(
            torch.tensor([1]),
        )

    for i in range(1, 5):
        assert beignet.polynomial.power_series_companion(
            torch.tensor([0] * i + [1]),
        ).shape == (i, i)

    assert beignet.polynomial.power_series_companion(torch.tensor([1, 2]))[0, 0] == -0.5
