import beignet.polynomial
import torch


def test_hermediv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.polynomial.hermediv(
                beignet.polynomial.hermeadd(
                    torch.tensor([0] * i + [1]),
                    torch.tensor([0] * j + [1]),
                ),
                torch.tensor([0] * i + [1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.hermetrim(
                    beignet.polynomial.hermeadd(
                        beignet.polynomial.hermemul(
                            quotient,
                            torch.tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.hermetrim(
                    beignet.polynomial.hermeadd(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
            )
