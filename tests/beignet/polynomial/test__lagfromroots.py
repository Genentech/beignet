import math

import torch
from beignet.polynomial import lag2poly, lagfromroots, lagtrim, lagval


def test_lagfromroots():
    torch.testing.assert_close(
        lagtrim(
            lagfromroots(
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

        output = lag2poly(
            lagfromroots(
                roots,
            ),
        )

        torch.testing.assert_close(
            output,
            torch.tensor([1.0]),
        )

        output = lagval(
            roots,
            lagfromroots(
                roots,
            ),
        )

        torch.testing.assert_close(
            output,
            torch.tensor([0.0]),
        )
