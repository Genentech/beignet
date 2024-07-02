import torch
from beignet.polynomial import polyadd, polydiv, polymul, polytrim


def test_polydiv():
    quotient, remainder = polydiv(
        torch.tensor([2.0]),
        torch.tensor([2.0]),
    )

    torch.testing.assert_close(
        quotient,
        torch.tensor([1.0]),
    )

    torch.testing.assert_close(
        remainder,
        torch.tensor([0.0]),
    )

    quotient, remainder = polydiv(
        torch.tensor([2.0, 2.0]),
        torch.tensor([2.0]),
    )

    torch.testing.assert_close(
        quotient,
        torch.tensor([1.0, 1.0]),
    )

    torch.testing.assert_close(
        remainder,
        torch.tensor([0.0]),
    )

    for j in range(5):
        for k in range(5):
            input = torch.tensor([0.0] * j + [1.0, 2.0])
            other = torch.tensor([0.0] * k + [1.0, 2.0])

            quotient, remainder = polydiv(
                polyadd(
                    input,
                    other,
                ),
                input,
            )

            torch.testing.assert_close(
                polytrim(
                    polyadd(
                        polymul(
                            quotient,
                            input,
                        ),
                        remainder,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    polyadd(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
            )
