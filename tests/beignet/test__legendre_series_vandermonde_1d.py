import beignet.polynomial
import pytest
import torch


def test_legendre_series_vandermonde_1d():
    input = torch.arange(3)

    output = beignet.polynomial.legendre_series_vandermonde_1d(input, 3)

    assert output.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            beignet.polynomial.evaluate_legendre_series_1d(
                input,
                torch.tensor([0] * index + [1]),
            ),
        )

    input = torch.tensor([[1, 2], [3, 4], [5, 6]])

    output = beignet.polynomial.legendre_series_vandermonde_1d(input, 3)

    assert output.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            beignet.polynomial.evaluate_legendre_series_1d(
                input,
                [0] * index + [1],
            ),
        )

    with pytest.raises(ValueError):
        beignet.polynomial.legendre_series_vandermonde_1d(
            torch.tensor([1, 2, 3]),
            -1,
        )
