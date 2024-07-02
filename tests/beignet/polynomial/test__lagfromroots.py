import math

import beignet.polynomial
import torch


def test_lagfromroots():
    torch.testing.assert_close(
        beignet.polynomial.lagtrim(
            beignet.polynomial.lagfromroots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for i in range(1, 5):
        roots = torch.linspace(-math.pi, 0, 2 * i + 1)

        roots = roots[1::2]

        roots = torch.cos(roots)

        output = beignet.polynomial.lag2poly(
            beignet.polynomial.lagfromroots(
                roots,
            ),
        )

        torch.testing.assert_close(
            output,
            torch.tensor([1.0]),
        )

        output = beignet.polynomial.lagval(
            roots,
            beignet.polynomial.lagfromroots(
                roots,
            ),
        )

        torch.testing.assert_close(
            output,
            torch.tensor([0.0]),
        )
