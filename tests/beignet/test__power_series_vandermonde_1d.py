import beignet.polynomial
import pytest
import torch.testing


def test_power_series_vandermonde_1d():
    x = torch.arange(3)

    v = beignet.polynomial.power_series_vandermonde_1d(x, 3)

    assert v.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            beignet.polynomial.evaluate_power_series_1d(
                x,
                torch.tensor([0] * index + [1]),
            ),
        )

    x = torch.tensor([[1, 2], [3, 4], [5, 6]])

    v = beignet.polynomial.power_series_vandermonde_1d(x, 3)

    assert v.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            v[..., index],
            beignet.polynomial.evaluate_power_series_1d(
                x,
                torch.tensor([0] * index + [1]),
            ),
        )

    with pytest.raises(ValueError):
        beignet.polynomial.power_series_vandermonde_1d(torch.arange(3), -1)
