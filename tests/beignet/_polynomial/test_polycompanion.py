import beignet.polynomial
import pytest
import torch


def test_polycompanion():
    with pytest.raises(ValueError):
        beignet.polynomial.polycompanion(
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        beignet.polynomial.polycompanion(
            torch.tensor([1]),
        )

    for i in range(1, 5):
        assert beignet.polynomial.polycompanion(
            torch.tensor([0] * i + [1]),
        ).shape == (i, i)

    assert beignet.polynomial.polycompanion(torch.tensor([1, 2]))[0, 0] == -0.5
