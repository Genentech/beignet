import math

import beignet.polynomial
import torch


def test_hermfromroots():
    torch.testing.assert_close(
        beignet.polynomial.hermtrim(
            beignet.polynomial.hermfromroots(
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
            beignet.polynomial.herm2poly(
                beignet.polynomial.hermfromroots(
                    roots,
                ),
            )[-1],
            torch.tensor([1.0]),
        )

        torch.testing.assert_close(
            beignet.polynomial.hermval(
                roots,
                beignet.polynomial.hermfromroots(
                    roots,
                ),
            ),
            target,
        )
