import beignet.polynomial
import torch.testing


def test_multiply_laguerre_series():
    x = torch.linspace(-3, 3, 100)

    for j in range(5):
        a = beignet.polynomial.evaluate_laguerre_series_1d(
            x,
            torch.tensor([0] * j + [1], dtype=torch.float64),
        )

        for k in range(5):
            b = beignet.polynomial.evaluate_laguerre_series_1d(
                x,
                torch.tensor([0] * k + [1], dtype=torch.float64),
            )

            c = beignet.polynomial.evaluate_laguerre_series_1d(
                x,
                beignet.polynomial.multiply_laguerre_series(
                    torch.tensor([0] * j + [1], dtype=torch.float64),
                    torch.tensor([0] * k + [1], dtype=torch.float64),
                ),
            )

            torch.testing.assert_close(c, a * b, atol=1e-5, rtol=1e-5)
