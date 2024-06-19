import beignet.polynomial
import beignet.polynomial._legadd
import beignet.polynomial._legdiv
import beignet.polynomial._legmul
import beignet.polynomial._legtrim
import torch


def test_legdiv():
    for i in range(5):
        for j in range(5):
            quotient, remainder = beignet.polynomial.legdiv(
                beignet.polynomial.legadd(
                    torch.tensor([0] * i + [1]),
                    torch.tensor([0] * j + [1]),
                ),
                torch.tensor([0] * i + [1]),
            )

            print(quotient)

            torch.testing.assert_close(
                beignet.polynomial.legtrim(
                    beignet.polynomial.legadd(
                        beignet.polynomial.legmul(
                            quotient,
                            torch.tensor([0] * i + [1]),
                        ),
                        remainder,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.legtrim(
                    beignet.polynomial.legadd(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=1e-6,
                ),
            )
