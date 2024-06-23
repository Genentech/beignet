import beignet.polynomial
import torch


def test_probabilists_hermite_series_vandermonde_1d():
    output = beignet.polynomial.probabilists_hermite_series_vandermonde_1d(
        torch.arange(3),
        3,
    )

    assert output.shape == (3, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            beignet.polynomial.evaluate_probabilists_hermite_series_1d(
                torch.arange(3),
                torch.tensor([0] * index + [1], dtype=torch.float32),
            ),
        )

    output = beignet.polynomial.probabilists_hermite_series_vandermonde_1d(
        torch.tensor([[1, 2], [3, 4], [5, 6]]),
        3,
    )

    assert output.shape == (3, 2, 4)

    for index in range(4):
        torch.testing.assert_close(
            output[..., index],
            beignet.polynomial.evaluate_probabilists_hermite_series_1d(
                torch.tensor([[1, 2], [3, 4], [5, 6]]),
                torch.tensor([0] * index + [1]),
            ),
        )
