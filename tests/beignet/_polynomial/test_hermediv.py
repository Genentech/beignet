import beignet.polynomial
import torch


def test_hermediv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.polynomial.divide_probabilists_hermite_series(
                beignet.polynomial.add_probabilists_hermite_series(
                    torch.tensor([0] * i + [1]),
                    torch.tensor([0] * j + [1]),
                ),
                torch.tensor([0] * i + [1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_probabilists_hermite_series(
                    beignet.polynomial.add_probabilists_hermite_series(
                        beignet.polynomial.multiply_probabilists_hermite_series(
                            quotient,
                            torch.tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_probabilists_hermite_series(
                    beignet.polynomial.add_probabilists_hermite_series(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
            )
