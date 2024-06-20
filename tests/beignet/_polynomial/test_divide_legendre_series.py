import beignet.polynomial
import torch


def test_divide_legendre_series():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.polynomial.divide_legendre_series(
                beignet.polynomial.add_legendre_series(
                    torch.tensor([0] * i + [1]),
                    torch.tensor([0] * j + [1]),
                ),
                torch.tensor([0] * i + [1]),
            )

            print(quotient)

            torch.testing.assert_close(
                beignet.polynomial.trim_legendre_series(
                    beignet.polynomial.add_legendre_series(
                        beignet.polynomial.multiply_legendre_series(
                            quotient,
                            torch.tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_legendre_series(
                    beignet.polynomial.add_legendre_series(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
            )
