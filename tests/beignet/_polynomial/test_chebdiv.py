import beignet.polynomial
import torch


def test_chebdiv():
    for i in range(5):
        for j in range(5):
            tgt = beignet.polynomial.add_chebyshev_series(
                torch.tensor([0] * i + [1]),
                torch.tensor([0] * j + [1]),
            )

            quotient, remainder = beignet.polynomial.divide_chebyshev_series(
                tgt,
                torch.tensor([0] * i + [1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_chebyshev_series(
                    beignet.polynomial.add_chebyshev_series(
                        beignet.polynomial.multiply_chebyshev_series(
                            quotient,
                            torch.tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_chebyshev_series(
                    tgt,
                    tolerance=1e-6,
                ),
            )
