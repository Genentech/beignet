import beignet.polynomial
import pytest
import torch


def test_legendre_series_companion():
    with pytest.raises(ValueError):
        beignet.polynomial.legendre_series_companion(
            torch.tensor([]),
        )

    with pytest.raises(ValueError):
        beignet.polynomial.legendre_series_companion(
            torch.tensor([1]),
        )

    for index in range(1, 5):
        output = beignet.polynomial.legendre_series_companion(
            torch.tensor([0] * index + [1]),
        )

        assert output.shape == (index, index)

    output = beignet.polynomial.legendre_series_companion(
        torch.tensor([1, 2]),
    )

    assert output[0, 0] == -0.5
