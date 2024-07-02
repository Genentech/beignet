import math

import beignet.polynomial
import torch


def test_legfromroots():
    torch.testing.assert_close(
        beignet.polynomial.legtrim(
            beignet.polynomial.legfromroots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for index in range(1, 5):
        input = torch.linspace(-math.pi, 0, 2 * index + 1)[1::2]

        output = beignet.polynomial.legfromroots(
            torch.cos(
                input,
            ),
        )

        assert output.shape[-1] == index + 1

        torch.testing.assert_close(
            beignet.polynomial.leg2poly(
                beignet.polynomial.legfromroots(
                    torch.cos(
                        input,
                    ),
                )
            )[-1],
            torch.tensor([1.0]),
        )

        torch.testing.assert_close(
            beignet.polynomial.legval(
                torch.cos(
                    input,
                ),
                beignet.polynomial.legfromroots(
                    torch.cos(
                        input,
                    ),
                ),
            ),
            torch.tensor([0.0]),
        )
