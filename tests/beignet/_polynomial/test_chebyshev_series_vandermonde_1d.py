import beignet.polynomial
import torch.testing


def test_chebyshev_series_vandermonde_1d():
    x = torch.arange(3)

    for index in range(4):
        torch.testing.assert_close(
            beignet.polynomial.chebyshev_series_vandermonde_1d(
                x,
                3,
            )[..., index],
            beignet.polynomial.evaluate_chebyshev_series_1d(
                x,
                torch.tensor([0] * index + [1], dtype=torch.float32),
            ),
        )

    x = torch.tensor([[1, 2], [3, 4], [5, 6]])

    for index in range(4):
        torch.testing.assert_close(
            beignet.polynomial.chebyshev_series_vandermonde_1d(
                x,
                3,
            )[..., index],
            beignet.polynomial.evaluate_chebyshev_series_1d(
                x,
                torch.tensor([0] * index + [1]),
            ),
        )
