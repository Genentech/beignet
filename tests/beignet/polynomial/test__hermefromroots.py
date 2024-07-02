import math

import beignet.polynomial
import torch


def test_hermefromroots():
    torch.testing.assert_close(
        beignet.polynomial.hermetrim(
            beignet.polynomial.hermefromroots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for i in range(1, 5):
        roots = torch.cos(torch.linspace(-math.pi, 0, 2 * i + 1)[1::2])

        pol = beignet.polynomial.hermefromroots(roots)

        assert len(pol) == i + 1

        torch.testing.assert_close(
            beignet.polynomial.herme2poly(pol)[-1],
            torch.tensor(1.0),
        )

        torch.testing.assert_close(
            beignet.polynomial.hermeval(roots, pol),
            torch.tensor([0.0]),
        )
