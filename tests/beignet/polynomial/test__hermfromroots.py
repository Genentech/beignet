import math

import torch
from beignet.polynomial import herm2poly, hermfromroots, hermtrim, hermval


def test_hermfromroots():
    torch.testing.assert_close(
        hermtrim(
            hermfromroots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for i in range(1, 5):
        roots = torch.cos(torch.linspace(-math.pi, 0, 2 * i + 1)[1::2])
        target = 0

        torch.testing.assert_close(
            herm2poly(
                hermfromroots(
                    roots,
                ),
            )[-1],
            torch.tensor([1.0]),
        )

        torch.testing.assert_close(
            hermval(
                roots,
                hermfromroots(
                    roots,
                ),
            ),
            target,
        )
