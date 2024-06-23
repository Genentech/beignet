import beignet.polynomial
import torch


def test_multiply_legendre_series():
    x = torch.linspace(-1, 1, 100)

    for j in range(5):
        a = beignet.polynomial.evaluate_legendre_series_1d(
            x,
            torch.tensor([0] * j + [1], dtype=torch.float64),
        )

        for k in range(5):
            b = beignet.polynomial.evaluate_legendre_series_1d(
                x,
                torch.tensor([0] * k + [1], dtype=torch.float64),
            )

            c = beignet.polynomial.evaluate_legendre_series_1d(
                x,
                beignet.polynomial.multiply_legendre_series(
                    torch.tensor([0] * j + [1], dtype=torch.float64),
                    torch.tensor([0] * k + [1], dtype=torch.float64),
                ),
            )

            torch.testing.assert_close(c, a * b)
