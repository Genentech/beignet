import beignet.polynomial
import torch


def test_polydiv():
    quotient, remainder = beignet.polynomial.divide_power_series(
        torch.tensor([2]),
        torch.tensor([2]),
    )

    torch.testing.assert_close(
        quotient,
        torch.tensor([1], dtype=torch.float64),
    )

    torch.testing.assert_close(
        remainder,
        torch.tensor([0], dtype=torch.float64),
    )

    quotient, remainder = beignet.polynomial.divide_power_series(
        torch.tensor([2, 2]),
        torch.tensor([2]),
    )

    torch.testing.assert_close(
        quotient,
        torch.tensor([1, 1], dtype=torch.float64),
    )

    torch.testing.assert_close(
        remainder,
        torch.tensor([0], dtype=torch.float64),
    )

    for j in range(5):
        for k in range(5):
            quotient, remainder = beignet.polynomial.divide_power_series(
                beignet.polynomial.add_power_series(
                    torch.tensor([0] * j + [1, 2]),
                    torch.tensor([0] * k + [1, 2]),
                ),
                torch.tensor([0] * j + [1, 2]),
            )

            torch.testing.assert_close(
                beignet.polynomial.add_power_series(
                    beignet.polynomial.multiply_power_series(
                        quotient,
                        torch.tensor([0] * j + [1, 2]),
                    ),
                    remainder,
                ),
                beignet.polynomial.add_power_series(
                    torch.tensor([0] * j + [1, 2]),
                    torch.tensor([0] * k + [1, 2]),
                ),
            )
