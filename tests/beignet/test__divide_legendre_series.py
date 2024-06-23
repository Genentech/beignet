import beignet.polynomial
import torch


def test_divide_legendre_series():
    for j in range(5):
        for k in range(5):
            quotient, remainder = beignet.polynomial.divide_legendre_series(
                beignet.polynomial.add_legendre_series(
                    torch.tensor([0] * j + [1]),
                    torch.tensor([0] * k + [1]),
                ),
                torch.tensor([0] * j + [1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_legendre_series(
                    beignet.polynomial.add_legendre_series(
                        beignet.polynomial.multiply_legendre_series(
                            quotient,
                            torch.tensor([0] * j + [1]),
                        ),
                        remainder,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_legendre_series(
                    beignet.polynomial.add_legendre_series(
                        torch.tensor([0] * j + [1]),
                        torch.tensor([0] * k + [1]),
                    ),
                    tolerance=1e-6,
                ),
            )
