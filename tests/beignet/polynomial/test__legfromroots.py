import math

import torch
from beignet.polynomial import leg2poly, legfromroots, legtrim, legval


def test_legfromroots():
    torch.testing.assert_close(
        legtrim(
            legfromroots(
                torch.tensor([]),
            ),
            tol=0.000001,
        ),
        torch.tensor([1.0]),
    )

    for index in range(1, 5):
        input = torch.linspace(-math.pi, 0, 2 * index + 1)[1::2]

        output = legfromroots(
            torch.cos(
                input,
            ),
        )

        assert output.shape[-1] == index + 1

        torch.testing.assert_close(
            leg2poly(
                legfromroots(
                    torch.cos(
                        input,
                    ),
                )
            )[-1],
            torch.tensor([1.0]),
        )

        torch.testing.assert_close(
            legval(
                torch.cos(
                    input,
                ),
                legfromroots(
                    torch.cos(
                        input,
                    ),
                ),
            ),
            torch.tensor([0.0]),
        )
