import beignet.polynomial
import torch


def test_chebdiv():
    for i in range(5):
        for j in range(5):
            tgt = beignet.polynomial.chebadd(
                torch.tensor([0] * i + [1]),
                torch.tensor([0] * j + [1]),
            )

            quotient, remainder = beignet.polynomial.chebdiv(
                tgt,
                torch.tensor([0] * i + [1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebadd(
                        beignet.polynomial.chebmul(
                            quotient,
                            torch.tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.chebtrim(
                    tgt,
                    tolerance=1e-6,
                ),
            )
