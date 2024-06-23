import beignet.polynomial
import torch


def test_divide_probabilists_hermite_series():
    for j in range(5):
        for k in range(5):
            quotient, remainder = beignet.polynomial.divide_probabilists_hermite_series(
                beignet.polynomial.add_probabilists_hermite_series(
                    torch.tensor([0] * j + [1]),
                    torch.tensor([0] * k + [1]),
                ),
                torch.tensor([0] * j + [1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_probabilists_hermite_series(
                    beignet.polynomial.add_probabilists_hermite_series(
                        beignet.polynomial.multiply_probabilists_hermite_series(
                            quotient,
                            torch.tensor([0] * j + [1]),
                        ),
                        remainder,
                    ),
                    tolerance=0.000001,
                ),
                beignet.polynomial.trim_probabilists_hermite_series(
                    beignet.polynomial.add_probabilists_hermite_series(
                        torch.tensor([0] * j + [1]),
                        torch.tensor([0] * k + [1]),
                    ),
                    tolerance=0.000001,
                ),
            )
